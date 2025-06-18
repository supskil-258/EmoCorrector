import matplotlib
matplotlib.use('Agg')
from tasks.tts.dataset_utils import FastSpeechDataset, BaseTTSDataset
import glob
import importlib
from utils.pitch_utils import clean_norm_interp_f0, clean_denorm_f0, f0_to_coarse
from tasks.tts.tts_utils import load_data_preprocessor
from data_gen.tts.emotion import inference as EmotionEncoder
from data_gen.tts.emotion.inference import embed_utterance as Embed_utterance
from data_gen.tts.emotion.inference import preprocess_wav
from tqdm import tqdm
from utils.hparams import hparams
from data_gen.tts.data_gen_utils import build_phone_encoder, build_word_encoder
import random
import torch
import torch.optim
import torch.utils.data
from utils.indexed_datasets import IndexedDataset
from resemblyzer import VoiceEncoder
import torch.distributions
import numpy as np
import utils

class EmoCorrector_dataset(BaseTTSDataset):
    def __init__(self, prefix, shuffle=False, test_items=None, test_sizes=None, data_dir=None):
        super().__init__(prefix, shuffle, test_items, test_sizes, data_dir)
        self.clean_f0_mean, self.clean_f0_std = hparams.get('clean_f0_mean', None), hparams.get('clean_f0_std', None)

        if prefix == 'valid':
            indexed_ds = IndexedDataset(f'{self.data_dir}/train')
            clean_sizes = np.load(f'{self.data_dir}/train_clean_lengths.npy')
            index = [i for i in range(len(indexed_ds))]
            random.shuffle(index)
            index = index[:300]
            self.clean_sizes = clean_sizes[index]
            self.indexed_ds = []
            for i in index:
                self.indexed_ds.append(indexed_ds[i])
            self.clean_avail_idxs = list(range(len(self.clean_sizes)))
            if hparams['min_frames'] > 0:
                self.clean_avail_idxs = [x for x in self.clean_avail_idxs if
                                         self.clean_sizes[x] >= hparams['min_frames']]
            self.clean_sizes = [self.clean_sizes[i] for i in self.clean_avail_idxs]

        if prefix == 'test' and hparams['test_input_dir'] != '':
            self.preprocessor, self.preprocess_args = load_data_preprocessor()
            self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            self.avail_idxs = [i for i, _ in enumerate(self.sizes)]

    def load_test_inputs(self, test_input_dir):
        inp_wav_paths = sorted(glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3'))
        binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizerr.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)

        phone_encoder = build_phone_encoder(hparams['binary_data_dir'])
        word_encoder = build_word_encoder(hparams['binary_data_dir'])
        voice_encoder = VoiceEncoder().cuda()

        encoder = [phone_encoder, word_encoder]
        sizes = []
        items = []
        EmotionEncoder.load_model(hparams['emotion_encoder_path'])
        preprocessor, preprocess_args = self.preprocessor, self.preprocess_args

        for wav_fn in tqdm(inp_wav_paths):
            item_name = wav_fn[len(test_input_dir) + 1:].replace("/", "_")
            spk_id = emotion = 0
            item2tgfn = wav_fn.replace('.wav', '.TextGrid') # prepare textgrid alignment
            txtpath = wav_fn.replace('.wav', '.txt')  # prepare text
            with open(txtpath, 'r') as f:
                text_raw = f.readlines()
                f.close()
            ph, txt = preprocessor.txt_to_ph(preprocessor.txt_processor, text_raw[0], preprocess_args)

            item = binarizer_cls.process_item(item_name, ph, txt, item2tgfn, wav_fn, spk_id, emotion, encoder, hparams['binarization_args'])
            item['emo_embed'] = Embed_utterance(preprocess_wav(item['wav_fn']))
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav'])
            items.append(item)
            sizes.append(item['len'])
        return items, sizes

    def _get_item(self, index):
        if hasattr(self, 'clean_avail_idxs') and self.clean_avail_idxs is not None:
            clean_index = self.clean_avail_idxs[index]

        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[clean_index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        checka = len(item['clean_mel'])  # 163
        checkb = self.clean_sizes[index]  # 163
        assert len(item['clean_mel']) == self.clean_sizes[index], (len(item['clean_mel']), self.clean_sizes[index])
        max_frames = hparams['max_frames']
        clean_spec = torch.Tensor(item['clean_mel'])[:max_frames]
        clean_max_frames = clean_spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        clean_spec = clean_spec[:clean_max_frames]
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,
            "clean_mel": clean_spec,
            "clean_mel_nonpadding": clean_spec.abs().sum(-1) > 0
        }
        clean_spec = sample['clean_mel']
        clean_T = clean_spec.shape[0]
        sample['clean_mel2ph'] = clean_mel2ph = torch.LongTensor(item['clean_mel2ph'])[
                                                :clean_T] if 'clean_mel2ph' in item else None

        if hparams['use_pitch_embed']:
            assert 'clean_f0' in item
            if hparams.get('normalize_pitch', False):
                clean_f0 = item["clean_f0"]
                if len(clean_f0 > 0) > 0 and clean_f0[clean_f0 > 0].std() > 0:
                    clean_f0[clean_f0 > 0] = (clean_f0[clean_f0 > 0] - clean_f0[clean_f0 > 0].mean()) / clean_f0[clean_f0 > 0].std() * hparams['clean_f0_std'] + \
                                 hparams['clean_f0_mean']
                    clean_f0[clean_f0 > 0] = clean_f0[clean_f0 > 0].clip(min=60, max=500)

                clean_pitch = f0_to_coarse(clean_f0)
                clean_pitch = torch.LongTensor(clean_pitch[:max_frames])
            else:
                clean_pitch = torch.LongTensor(item.get("clean_pitch"))[:max_frames] if "clean_pitch" in item else None
            clean_f0, clean_uv = clean_norm_interp_f0(item["clean_f0"][:max_frames], hparams)
            clean_uv = torch.FloatTensor(clean_uv)
            clean_f0 = torch.FloatTensor(clean_f0)
        else:
            clean_f0 = clean_uv = torch.zeros_like(clean_mel2ph)
            clean_pitch = None
        sample["clean_f0"], sample["clean_uv"], sample["clean_pitch"] = clean_f0, clean_uv, clean_pitch

        sample["emotion_id"] = item['emotion']  # emotion id
        sample["spk_id"] = item['spk_id']  # speaker id
        sample["clean_wav_fn"] = item['clean_wav_fn']
        sample["emo_embedding"] = torch.Tensor(item['emo_embedding'])

        if hparams.get('use_word', False):
            sample["ph_words"] = item["ph_words"]
            sample["word_tokens"] = torch.LongTensor(item["word_tokens"])
            sample["clean_mel2word"] = torch.LongTensor(item.get("clean_mel2word"))[:max_frames]
            sample["ph2word"] = torch.LongTensor(item['ph2word'][:hparams['max_input_tokens']])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
        clean_mels = utils.collate_2d([s['clean_mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        clean_mel_lengths = torch.LongTensor([s['clean_mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'clean_mels': clean_mels,
            'clean_mel_lengths': clean_mel_lengths
        }

        clean_f0 = utils.collate_1d([s['clean_f0'] for s in samples], 0.0)
        clean_pitch = utils.collate_1d([s['clean_pitch'] for s in samples]) if samples[0][
                                                                                   'clean_pitch'] is not None else None
        clean_uv = utils.collate_1d([s['clean_uv'] for s in samples])

        clean_mel2ph = utils.collate_1d([s['clean_mel2ph'] for s in samples], 0.0) if samples[0][
                                                                                          'clean_mel2ph'] is not None else None
        batch.update({
            'clean_mel2ph': clean_mel2ph,
            'clean_pitch': clean_pitch,
            'clean_f0': clean_f0,
            'clean_uv': clean_uv
        })
        batch['clean_wav_fn'] = [s['clean_wav_fn'] for s in samples]
        emo_embedding = torch.stack([s['emo_embedding'] for s in samples])
        batch['emo_embedding'] = emo_embedding

        if hparams.get('use_word', False):
            ph_words = [s['ph_words'] for s in samples]
            batch['ph_words'] = ph_words
            word_tokens = utils.collate_1d([s['word_tokens'] for s in samples], 0)
            batch['word_tokens'] = word_tokens
            clean_mel2word = utils.collate_1d([s['clean_mel2word'] for s in samples], 0)
            batch['clean_mel2word'] = clean_mel2word
            ph2word = utils.collate_1d([s['ph2word'] for s in samples], 0)
            batch['ph2word'] = ph2word

        # 聚合 spk_id 和 emotion_id
        spk_id = [s['spk_id'] for s in samples]
        batch['spk_id'] = spk_id

        emotion_id = [s['emotion_id'] for s in samples]
        batch['emotion_id'] = emotion_id


        return batch

