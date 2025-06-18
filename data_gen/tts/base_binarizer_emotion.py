import os
os.environ["OMP_NUM_THREADS"] = "1"
from collections import Counter
from utils.text_encoder import TokenTextEncoder
from utils.multiprocess_utils import chunked_multiprocess_run
from resemblyzer import VoiceEncoder
import random
import traceback
import json
from tqdm import tqdm
from data_gen.tts.data_gen_utils import get_mel2ph, get_pitch, build_phone_encoder, is_sil_phoneme
from utils.hparams import hparams, set_hparams
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder
from vocoders.base_vocoder import get_vocoder_cls
import pandas as pd



class BinarizationError(Exception):
    pass


class EmotionBinarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.pre_align_args = hparams['pre_align_args']
        self.emo_embedding_dir = hparams['emo_embedding_dir']
        self.item2txt = {}
        self.item2ph = {}
        self.item2clean_wavfn = {}
        self.item2clean_tgfn = {}
        self.item2spk = {}
        self.item2emo = {}

    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            self.meta_df = pd.read_csv(f"{processed_data_dir}/metadata_phone.csv", dtype=str)
            for r_idx, r in tqdm(self.meta_df.iterrows(), desc='Loading meta data.'):
                item_name = raw_item_name = r['item_name']
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2txt[item_name] = r['txt']
                self.item2ph[item_name] = r['ph']
                self.item2clean_wavfn[item_name] = r['clean_wav_fn']
                self.item2spk[item_name] = r.get('spk_name', 'SPK1') \
                    if self.binarization_args['with_spk_id'] else 'SPK1'
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
                self.item2clean_tgfn[item_name] = f"{processed_data_dir}/mfa_outputs/clean/{raw_item_name}.TextGrid"
                self.item2emo[item_name] = r.get('emotion')
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        return self.item_names[hparams['test_num']+hparams['valid_num']:]

    @property
    def valid_item_names(self):
        return self.item_names[0: hparams['test_num']+hparams['valid_num']]  #

    @property
    def test_item_names(self):
        return self.item_names[0: hparams['test_num']]

    def build_spk_map(self):
        spk_map = set()
        for item_name in self.item_names:
            spk_name = self.item2spk[item_name]
            spk_map.add(spk_name)
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
        print("| #Spk: ", len(spk_map))
        assert len(spk_map) == 0 or len(spk_map) <= hparams['num_spk'], len(spk_map)
        return spk_map

    def build_emo_map(self):
        emo_map = set()
        for item_name in self.item_names:
            emo_name = self.item2emo[item_name]
            emo_map.add(emo_name)
        emo_map = {x: i for i, x in enumerate(sorted(list(emo_map)))}
        print("| #Emo: ", len(emo_map))
        return emo_map

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.item2spk[item_name]]

    def item_name2emo_id(self, item_name):
        return self.emo_map[self.item2emo[item_name]]

    def _phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = []
        if self.binarization_args['reset_phone_dict'] or not os.path.exists(ph_set_fn):
            for ph_sent in self.item2ph.values():
                ph_set += ph_sent.split(' ')
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'))
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
            print("| Load phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])


    def _word_encoder(self):
        fn = f"{hparams['binary_data_dir']}/word_set.json"
        word_set = []
        if self.binarization_args['reset_word_dict']:
            for word_sent in self.item2txt.values():
                word_set += [x for x in word_sent.split(' ') if x != '']
            word_set = Counter(word_set)
            total_words = sum(word_set.values())
            word_set = word_set.most_common(hparams['word_size'])
            num_unk_words = total_words - sum([x[1] for x in word_set])
            word_set = [x[0] for x in word_set]
            json.dump(word_set, open(fn, 'w'))
            print(f"| Build word set. Size: {len(word_set)}, #total words: {total_words},"
                  f" #unk_words: {num_unk_words}, word_set[:10]:, {word_set[:10]}.")
        else:
            word_set = json.load(open(fn, 'r'))
            print("| Load word set. Size: ", len(word_set), word_set[:10])
        return TokenTextEncoder(None, vocab_list=word_set, replace_oov='<UNK>')

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            ph = self.item2ph[item_name]
            txt = self.item2txt[item_name]
            clean_tg_fn = self.item2clean_tgfn.get(item_name)
            clean_wav_fn = self.item2clean_wavfn[item_name]
            spk_id = self.item_name2spk_id(item_name)
            emotion = self.item_name2emo_id(item_name)
            yield item_name, ph, txt, clean_tg_fn, clean_wav_fn, spk_id, emotion

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w'))

        self.emo_map = self.build_emo_map()
        print("| emo_map: ", self.emo_map)
        emo_map_fn = f"{hparams['binary_data_dir']}/emo_map.json"
        json.dump(self.emo_map, open(emo_map_fn, 'w'))

        self.phone_encoder = self._phone_encoder()
        self.word_encoder = None

        if self.binarization_args['with_word']:
            self.word_encoder = self._word_encoder()
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        ph_lengths = []
        clean_mel_lengths = []
        clean_f0s = []
        clean_total_sec = 0
        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()

        meta_data = list(self.meta_data(prefix))
        for m in meta_data:
            args.append(list(m) + [(self.phone_encoder, self.word_encoder), self.binarization_args])
        num_workers = self.num_workers
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
            if item is None:
                continue

            if not self.binarization_args['with_wav'] and 'clean_wav' in item:
                del item['clean_wav']
            if not self.binarization_args['with_wav'] and 'noisy_wav' in item:
                del item['noisy_wav']

            item_name = item['item_name']
            emo_embedding_path = os.path.join(hparams['emo_embedding_dir'], f"{item_name}.npy")
            if os.path.exists(emo_embedding_path):
                emo_embedding = np.load(emo_embedding_path)
                item['emo_embedding'] = emo_embedding.astype(np.float32)
            else:
                print(f"| Emotion embedding file not found: {emo_embedding_path}, skipping this item.")
                continue

            builder.add_item(item)
            clean_mel_lengths.append(item['clean_len'])
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            clean_total_sec += item['clean_sec']
            if item.get('clean_f0') is not None:
                clean_f0s.append(item['clean_f0'])
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_clean_lengths.npy', clean_mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)

        if len(clean_f0s) > 0:
            clean_f0s = np.concatenate(clean_f0s, 0)
            clean_f0s = clean_f0s[clean_f0s != 0]
            np.save(f'{data_dir}/{prefix}_clean_f0s_mean_std.npy', [np.mean(clean_f0s).item(), np.std(clean_f0s).item()])

        print(f"| {prefix} total duration: {clean_total_sec:.3f}s")

    @classmethod
    def process_item(cls, item_name, ph, txt, clean_tg_fn, clean_wav_fn, spk_id, emotion, encoder, binarization_args):
        res = {'item_name': item_name, 'txt': txt, 'ph': ph, 'clean_wav_fn': clean_wav_fn,'spk_id': spk_id, 'emotion': emotion}

        if binarization_args['with_linear']:
            clean_wav, clean_mel, clean_linear_stft = get_vocoder_cls(hparams).wav2spec(clean_wav_fn)
            res['clean_linear'] = clean_linear_stft
        else:
            clean_wav, clean_mel = get_vocoder_cls(hparams).wav2spec(clean_wav_fn)
        clean_wav = clean_wav.astype(np.float16)
        res.update({'clean_mel': clean_mel, 'clean_wav': clean_wav,
                    'clean_sec': len(clean_wav) / hparams['audio_sample_rate'],
                    'clean_len': clean_mel.shape[0]})

        try:
            if binarization_args['with_f0']:
                cls.get_pitch(res)
                if binarization_args['with_f0cwt']:
                    cls.get_f0cwt(res)
            if binarization_args['with_txt']:
                ph_encoder, word_encoder = encoder
                try:
                    res['phone'] = ph_encoder.encode(ph)
                    res['ph_len'] = len(res['phone'])
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(clean_tg_fn, res)
                    if binarization_args['trim_eos_bos']:
                        bos_dur = res['dur'][0]
                        eos_dur = res['dur'][-1]
                        res['clean_mel'] = clean_mel[bos_dur:-eos_dur]
                        res['clean_f0'] = res['clean_f0'][bos_dur:-eos_dur]
                        res['clean_pitch'] = res['clean_pitch'][bos_dur:-eos_dur]
                        res['clean_mel2ph'] = res['clean_mel2ph'][bos_dur:-eos_dur]
                        res['clean_wav'] = clean_wav[bos_dur * hparams['hop_size']:-eos_dur * hparams['hop_size']]
                        res['clean_dur'] = res['clean_dur'][1:-1]
                        res['clean_len'] = res['clean_mel'].shape[0]
                if binarization_args['with_word']:
                    cls.get_word(res, word_encoder)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, clean_wav_fn: {clean_wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, clean_wav_fn: {clean_wav_fn}")
            return None
        return res

    @staticmethod
    def get_align(clean_tg_fn, res):
        ph = res['ph']
        clean_mel = res['clean_mel']
        # noisy_mel = res['noisy_mel']
        phone_encoded = res['phone']
        if clean_tg_fn is not None and os.path.exists(clean_tg_fn):
            clean_mel2ph, clean_dur = get_mel2ph(clean_tg_fn, ph, clean_mel, hparams)
        else:
            raise BinarizationError(f"clean Align not found")

        if clean_mel2ph.max() - 1 >= len(phone_encoded):
            raise BinarizationError(
                f"clean Align does not match: mel2ph.max() - 1: {clean_mel2ph.max() - 1}, len(phone_encoded): {len(phone_encoded)}")

        res['clean_mel2ph'] = clean_mel2ph
        res['clean_dur'] = clean_dur

    @staticmethod
    def get_pitch(res):
        clean_wav, clean_mel = res['clean_wav'], res['clean_mel']
        clean_f0, clean_pitch_coarse = get_pitch(clean_wav, clean_mel, hparams)
        if sum(clean_f0) == 0:
            raise BinarizationError("Empty clean_f0")
        res['clean_f0'] = clean_f0
        res['clean_pitch'] = clean_pitch_coarse

    @staticmethod
    def get_f0cwt(res):
        from utils.cwt import get_cont_lf0, get_lf0_cwt
        clean_f0 = res['clean_f0']
        clean_uv, clean_cont_lf0_lpf = get_cont_lf0(clean_f0)
        clean_logf0s_mean_org, clean_logf0s_std_org = np.mean(clean_cont_lf0_lpf), np.std(clean_cont_lf0_lpf)
        clean_cont_lf0_lpf_norm = (clean_cont_lf0_lpf - clean_logf0s_mean_org) / clean_logf0s_std_org
        clean_Wavelet_lf0, clean_scales = get_lf0_cwt(clean_cont_lf0_lpf_norm)
        if np.any(np.isnan(clean_Wavelet_lf0)):
            raise BinarizationError("NaN clean_CWT")
        res['clean_cwt_spec'] = clean_Wavelet_lf0
        res['clean_cwt_scales'] = clean_scales
        res['clean_f0_mean'] = clean_logf0s_mean_org
        res['clean_f0_std'] = clean_logf0s_std_org

    @staticmethod
    def get_word(res, word_encoder):
        ph_split = res['ph'].split(" ")
        # ph side mapping to word
        ph_words = []  # ['<BOS>', 'N_AW1_', ',', 'AE1_Z_|', 'AO1_L_|', 'B_UH1_K_S_|', 'N_AA1_T_|', ....]
        ph2word = np.zeros([len(ph_split)], dtype=int)
        last_ph_idx_for_word = []  # [2, 11, ...]
        for i, ph in enumerate(ph_split):
            if ph == '|':
                last_ph_idx_for_word.append(i)
            elif not ph[0].isalnum():
                if ph not in ['<BOS>']:
                    last_ph_idx_for_word.append(i - 1)
                last_ph_idx_for_word.append(i)
        start_ph_idx_for_word = [0] + [i + 1 for i in last_ph_idx_for_word[:-1]]
        for i, (s_w, e_w) in enumerate(zip(start_ph_idx_for_word, last_ph_idx_for_word)):
            ph_words.append(ph_split[s_w:e_w + 1])
            ph2word[s_w:e_w + 1] = i
        ph2word = ph2word.tolist()
        ph_words = ["_".join(w) for w in ph_words]

        # mel side mapping to word
        clean_mel2word = []
        clean_dur_word = [0 for _ in range(len(ph_words))]
        for clean_i, clean_m2p in enumerate(res['clean_mel2ph']):
            clean_word_idx = ph2word[clean_m2p - 1]
            clean_mel2word.append(ph2word[clean_m2p - 1])
            clean_dur_word[clean_word_idx] += 1

        ph2word = [x + 1 for x in ph2word]
        clean_mel2word = [x + 1 for x in clean_mel2word]
        # noisy_mel2word = [x + 1 for x in noisy_mel2word]
        res['ph_words'] = ph_words  # [T_word]
        res['ph2word'] = ph2word  # [T_ph]
        res['clean_mel2word'] = clean_mel2word  # [T_mel]
        res['clean_dur_word'] = clean_dur_word  # [T_word]
        words = [x for x in res['txt'].split(" ") if x != '']
        while len(words) > 0 and is_sil_phoneme(words[0]):
            words = words[1:]
        while len(words) > 0 and is_sil_phoneme(words[-1]):
            words = words[:-1]
        words = ['<BOS>'] + words + ['<EOS>']
        word_tokens = word_encoder.encode(" ".join(words))
        res['words'] = words
        res['word_tokens'] = word_tokens
        assert len(words) == len(ph_words), [words, ph_words]

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))


if __name__ == "__main__":
    set_hparams()
    EmotionBinarizer().process()
