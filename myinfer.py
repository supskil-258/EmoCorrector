import os
import subprocess
import traceback
import torch
import numpy as np
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams
from utils.ckpt_utils import load_ckpt
from vocoders.hifigan import HifiGanGenerator
from modules.EmoCorrector.model.EmoCorrector_infer import EmoCorrector
from data_gen.tts.data_gen_utils import build_phone_encoder, get_pitch, get_mel2ph
from data_gen.tts.emotion.inference import preprocess_wav
from resemblyzer import VoiceEncoder
from data_gen.tts.txt_processors.base_text_processor import get_txt_processor_cls
from vocoders.base_vocoder import VOCODERS
import shutil
from funasr import AutoModel
os.environ["OMP_NUM_THREADS"] = "1"

def is_sil_phoneme(p):
    return not p[0].isalpha()
def read_txt_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            print(f"Read text content from {filepath}: {content}")
            return content
    except Exception as e:
        print(f"Failed to read file {filepath}: {e}")
        return None
def txt_to_ph(txt_processor, txt_raw, preprocess_args):
    txt_struct, txt = txt_processor.process(txt_raw, preprocess_args)
    ph = [p for w in txt_struct for p in w[1]]
    ph_gb_word = ["_".join(w[1]) for w in txt_struct]
    words = [w[0] for w in txt_struct]
    ph2word = [w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
    return " ".join(ph), txt, " ".join(words), ph2word, " ".join(ph_gb_word)
hp = set_hparams(print_hparams=False)
txt_processor_class = get_txt_processor_cls(hparams['preprocess_args']['txt_processor'])
txt_processor = txt_processor_class()
n_mels = 80
n_fft = hp['fft_size']
hop_length = hp['hop_size']
win_length = hp['win_size']
fmin = hp['fmin']
fmax = hp['fmax']
sr = hparams['audio_sample_rate']
def load_wav_to_torch(wav_path, sr=22050):
    if hparams['vocoder'] in VOCODERS:
        wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_path)
    else:
        wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_path)
    return torch.from_numpy(wav).float(), mel
def mel_spectrogram(wav, mel, sr=16000):
    print(f"mel shape: {mel.shape}")
    return mel
def extract_f0(wav, mel, sr):
    f0, pitch_coarse = get_pitch(wav.numpy(), mel, hparams)
    print(f"Extracted f0, shape: {f0.shape}, pitch_coarse shape: {pitch_coarse.shape}")
    if sum(f0) == 0:
        raise ValueError("F0 extraction is empty, there might be an issue with the audio")
    return f0, pitch_coarse
def extract_phonemes_from_text(txt_raw):
    ph, txt, words, ph2word, ph_gb_word = txt_to_ph(txt_processor, txt_raw, hparams['preprocess_args'])
    print(f"ph: {ph}, words: {words}, ph2word: {ph2word}")
    return ph, ph2word, ph_gb_word
def read_phoneme_alignment_from_textgrid(tg_fn):
    from praatio import textgrid
    tg = textgrid.openTextgrid(tg_fn, includeEmptyIntervals=True)
    tier_name = tg.tierNameList[0]
    intervals = tg.tierDict[tier_name].entryList
    tg_align = []
    for idx, interval in enumerate(intervals):
        tg_align.append({
            'idx': str(idx + 1),
            'xmin': interval.start,
            'xmax': interval.end,
            'text': interval.label
        })
    return tg_align
def extract_ref_mel2word(tg_fn, phonemes_str, mel_spectrogram_output, ph2word, ph_gb_word):
    tg_align = read_phoneme_alignment_from_textgrid(tg_fn)
    ph_list = phonemes_str.strip().split()
    special_symbols = {'<BOS>', '<EOS>', '|'}
    ph_list_clean = [p for p in ph_list if p not in special_symbols]
    tg_phonemes_raw = [entry['text'] for entry in tg_align if entry['text'] != '']
    tg_phonemes = []
    for phoneme_group in tg_phonemes_raw:
        tg_phonemes.extend(phoneme_group.split('_'))
    print(f"ph_list length: {len(ph_list_clean)}, tg_phonemes length: {len(tg_phonemes)}")
    print(f"ph_list: {ph_list_clean}")
    print(f"tg_phonemes: {tg_phonemes}")
    if len(ph_list_clean) != len(tg_phonemes):
        raise ValueError("The length of ph_list and tg_phonemes does not match, alignment is not possible")
    ref_mel2word = [0] * mel_spectrogram_output.shape[1]
    return torch.tensor(ref_mel2word).long().unsqueeze(0).cuda()
def extract_audio_features_with_mel2word(audio_path, txt_raw, ph_encoder, tg_fn):
    wav, mel = load_wav_to_torch(audio_path)
    mel_spectrogram_output = mel_spectrogram(wav, mel)
    mel_spectrogram_output = mel_spectrogram_output.astype(np.float16)
    f0, pitch_coarse = extract_f0(wav, mel_spectrogram_output, sr)
    wav_preprocessed = preprocess_wav(wav.cpu().numpy(), sr)
    voice_encoder = VoiceEncoder()
    spk_embed = voice_encoder.embed_utterance(wav_preprocessed)
    phonemes_str, ph2word, ph_gb_word = extract_phonemes_from_text(txt_raw)
    txt_tokens = ph_encoder.encode(phonemes_str)
    txt_tokens_tensor = torch.LongTensor(txt_tokens).unsqueeze(0).cuda()
    mel2ph, dur = get_mel2ph(tg_fn, phonemes_str, mel_spectrogram_output, hparams)
    mel2ph = torch.from_numpy(mel2ph).long().unsqueeze(0).cuda()
    ref_mel2word_tensor = extract_ref_mel2word(tg_fn, phonemes_str, mel_spectrogram_output, ph2word, ph_gb_word)
    mel_duration_sec = mel_spectrogram_output.shape[1] * hop_length / sr
    mel_length = mel_spectrogram_output.shape[1]

    return {
        'mel': mel_spectrogram_output,
        'f0': f0,
        'pitch_coarse': pitch_coarse,
        'spk_embed': spk_embed,
        # 'emo_embed': emo_embed,
        'txt_tokens': txt_tokens_tensor,
        'mel2ph': mel2ph,
        'ref_mel2word': ref_mel2word_tensor,
        'mel_duration_sec': mel_duration_sec,
        'mel_length': mel_length
    }

ph_set_path = "dataset/processed/Editing_emo_data16k"
ph_encoder = build_phone_encoder(ph_set_path)

# Define audio and text path
voice_audio_path = "assets/001_5_000234.wav"
voice_txt_path = "assets/010_6_000234.txt"

BASE_DIR = "assets/mfa_temp"
mfa_outputs = os.path.join(BASE_DIR, "mfa_outputs")
voice_txt_raw = read_txt_from_file(voice_txt_path)
input_dir = os.path.join(BASE_DIR, "mfa_inputs")
os.makedirs(input_dir, exist_ok=True)

def build_mfa_inputs(cls, item, clean_mfa_input_dir, mfa_group, clean_wav_processed_tmp, preprocess_args):
    item_name = item['item_name']
    clean_wav_align_fn = item['clean_wav_align_fn']
    ph_gb_word = item['ph_gb_word']
    ext = os.path.splitext(clean_wav_align_fn)[1]
    clean_mfa_input_group_dir = os.path.join(clean_mfa_input_dir, mfa_group)
    os.makedirs(clean_mfa_input_group_dir, exist_ok=True)
    clean_new_wav_align_fn = os.path.join(clean_mfa_input_group_dir, f"{item_name}{ext}")
    clean_move_link_func = shutil.move if os.path.dirname(clean_wav_align_fn) == clean_wav_processed_tmp else shutil.copy
    clean_move_link_func(clean_wav_align_fn, clean_new_wav_align_fn)
    ph_gb_word_nosil = " ".join([
        "_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
        for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)
    ])
    lab_file_path = os.path.join(clean_mfa_input_group_dir, f"{item_name}.lab")
    with open(lab_file_path, 'w') as f_txt:
        f_txt.write(ph_gb_word_nosil)
    return ph_gb_word_nosil, lab_file_path

class MFAHelper:
    @classmethod
    def build_mfa_inputs(cls, item, clean_mfa_input_dir, mfa_group, clean_wav_processed_tmp, preprocess_args):
        return build_mfa_inputs(cls, item, clean_mfa_input_dir, mfa_group, clean_wav_processed_tmp, preprocess_args)

items = [
    # {
    #     'item_name': os.path.splitext(os.path.basename(emotion_audio_path))[0],
    #     'clean_wav_align_fn': emotion_audio_path,
    #     'ph_gb_word': extract_phonemes_from_text(emotion_txt_raw)[2]
    # },
    {
        'item_name': os.path.splitext(os.path.basename(voice_audio_path))[0],
        'clean_wav_align_fn': voice_audio_path,
        'ph_gb_word': extract_phonemes_from_text(voice_txt_raw)[2]
    }
]

mfa_group = "default_group"
clean_wav_processed_tmp = "assets/mfa_temp/processed"
os.makedirs(clean_wav_processed_tmp, exist_ok=True)

for item in items:
    ph_gb_word_nosil, lab_file_path = MFAHelper.build_mfa_inputs(
        item=item,
        clean_mfa_input_dir=input_dir,
        mfa_group=mfa_group,
        clean_wav_processed_tmp=clean_wav_processed_tmp,
        preprocess_args=hparams['preprocess_args']
    )
    print(f"已处理项目 {item['item_name']}，生成的 .lab 文件: {lab_file_path}")

mfa_script_path = "mfa_align.sh"
print(f"正在运行 MFA 对齐脚本: {mfa_script_path}")
try:
    subprocess.run(['bash', mfa_script_path], check=True)
    print("MFA 对齐脚本执行成功。")
except subprocess.CalledProcessError as e:
    print(f"MFA 对齐脚本执行失败: {e}")
    traceback.print_exc()
    exit(1)

def get_aligned_textgrid(audio_path):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    tg_path = os.path.join(mfa_outputs, f"{base_name}.TextGrid")
    if not os.path.exists(tg_path):
        raise FileNotFoundError(f"未找到对齐后的 TextGrid 文件: {tg_path}")
    return tg_path
voice_tg_path = get_aligned_textgrid(voice_audio_path)
try:
    # emotion_source_features = extract_audio_features_with_mel2word(
    #     emotion_audio_path, emotion_txt_raw, ph_encoder, emotion_tg_path)
    voice_target_features = extract_audio_features_with_mel2word(
        voice_audio_path, voice_txt_raw, ph_encoder, voice_tg_path)
except Exception as e:
    print(f"特征提取失败: {e}")
    traceback.print_exc()
    exit(1)

tts = EmoCorrector(ph_encoder)
tts.eval()
state_dict = torch.load("checkpoints/EmoCorrector/model_ckpt_steps_300000.ckpt")['state_dict']['model']
device = torch.device('cuda')
tts.load_state_dict(state_dict)
tts.to(device)

#Put correct emotion audio path, it can be more then one
topk_sample = [
    "processed/Editing_emo_data16k/wav_processed/clean/003_6_000126.wav",
    "processed/Editing_emo_data16k/wav_processed/clean/002_6_000234.wav",
    "processed/Editing_emo_data16k/wav_processed/clean/001_6_000234.wav"]

CLAPspeech_encoder = AutoModel(model="iic/emotion2vec_plus_large")
rec_results = CLAPspeech_encoder.generate(topk_sample, output_dir="./outputs", granularity="utterance",
                                          extract_embedding=True)
all_emo_embeds = [rec_result['feats'] for rec_result in rec_results]
all_emo_embeds_np = np.array(all_emo_embeds)
emo_embedding = torch.tensor(all_emo_embeds_np, dtype=torch.float32)
emo_embedding = emo_embedding.mean(dim=0, keepdim=True)

try:
    with torch.no_grad():

        output = tts(
            txt_tokens = voice_target_features['txt_tokens'],
            spk_id=[0],
            emotion_id=[4],
            topk_sample=topk_sample,
            mel2ph=voice_target_features['mel2ph'],
            edit_mel2ph=voice_target_features['mel2ph'],
            ref_mel2word=voice_target_features['ref_mel2word'],
            edit_ref_mel2word=voice_target_features['ref_mel2word'],
            emo_embedding=emo_embedding,
            spk_ref_mel=torch.from_numpy(voice_target_features['mel']).float().unsqueeze(0).cuda(),
            global_steps=1000000,
            infer=True
    )
except Exception as e:
    print(f"模型推理失败: {e}")
    traceback.print_exc()
    exit(1)

hifigan_configpath = 'checkpoints/trainset_hifigan/config.yaml'
vocoder = HifiGanGenerator(set_hparams(hifigan_configpath, global_hparams=False))
load_ckpt(vocoder, 'checkpoints/trainset_hifigan/model_ckpt_steps_1000000.ckpt', 'model_gen')
vocoder.to(device)

try:
    vocoderin = output['mel_out'].transpose(2, 1).cuda()
    wav_out = vocoder(vocoderin)[:, 0].squeeze().cpu().detach().numpy()
    save_wav(wav_out, 'infer_out/emotion_transferred_output.wav', hp['audio_sample_rate'])
    print("音频生成并保存成功: infer_out/emotion_transferred_output.wav")
except Exception as e:
    print(f"Vocoder 处理失败: {e}")
    traceback.print_exc()
    exit(1)
