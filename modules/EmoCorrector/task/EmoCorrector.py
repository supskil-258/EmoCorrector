import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import os
from data_gen.tts.data_gen_utils import get_pitch
from modules.fastspeech.tts_modules import mel2ph_to_dur
from utils import audio
from utils.pitch_utils import clean_denorm_f0
from vocoders.base_vocoder import get_vocoder_cls
from utils.plot import spec_to_figure
from utils.hparams import hparams
from utils.tts_utils import select_attn
import utils
from modules.EmoCorrector.task.dataset import EmoCorrector_dataset
from modules.EmoCorrector.model.EmoCorrector import EmoCorrector
from tasks.tts.fs2 import FastSpeech2Task


class EmoCorrectorTask(FastSpeech2Task):
    def __init__(self):
        super(EmoCorrectorTask, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_cls = EmoCorrector_dataset


    def build_tts_model(self):
        self.model = EmoCorrector(self.phone_encoder)

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.num_params(self.model)
        return self.model


    def run_model(self, model, batch_index, sample, return_output=False):
        # 提取必要的数据
        text = sample['text']
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['clean_mels']      # [B, T_s, 80]
        mel2ph = sample['clean_mel2ph']    # [B, T_s]
        mel2word = sample['clean_mel2word']
        clean_f0 = sample['clean_f0']      # [B, T_s]
        clean_uv = sample['clean_uv']      # [B, T_s] 0/1
        clean_wav_fn = sample['clean_wav_fn']
        spk_id = sample['spk_id']
        emotion_id = sample['emotion_id']
        emo_embedding = sample['emo_embedding']


        # 定义损失函数
        losses = {}

        output = model(
            text = text,
            txt_tokens=txt_tokens,
            spk_id=spk_id,
            emotion_id=emotion_id,
            mel2ph=mel2ph,
            ref_mel2ph=mel2ph,
            ref_mel2word=mel2word,
            spk_ref_mel=target,
            # emo_ref_mel=target,
            emo_embedding=emo_embedding,
            clean_wav_fn=clean_wav_fn,
            f0=clean_f0,
            uv=clean_uv,
            global_steps=self.global_step,
            infer=False,
            ckeck=False,

        )
        if self.global_step < 200000:
            losses['loss_CLAP'] = output['loss_CLAP']

        # teacher model loss
        losses['clean_loss_emo2emo'] = output['clean_loss_emo2emo']
        losses['clean_loss_spk2spk'] = output['clean_loss_spk2spk']

        # disentangling
        losses['clean_loss_grl_spk2emo'] = output['clean_loss_grl_spk2emo']
        losses['clean_loss_grl_emo2spk'] = output['clean_loss_grl_emo2spk']

        #20000
        if self.global_step > 20000:

            # tts
            losses['postflow'] = output['postflow']
            self.add_mel_loss(output['mel_out'], target, losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)


        return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'], model_out = self.run_model(self.model, batch_idx, sample, return_output=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        # encdec_attn = model_out['select_attn']
        if self.global_step > 20000:
            mel_out = self.model.out2mel(model_out['mel_out'])
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 and self.global_step>20000:
            vmin = hparams['mel_vmin']
            vmax = hparams['mel_vmax']
            # 需要修改
            self.plot_mel(batch_idx, sample['clean_mels'], mel_out)
            self.plot_dur(batch_idx, sample, model_out)
            if hparams['use_pitch_embed']:
                self.plot_pitch(batch_idx, sample, model_out)
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            if self.global_step > 10000:
                text = sample['text']
                txt_tokens = sample['txt_tokens']  # [B, T_t]
                target = sample['clean_mels']  # [B, T_s, 80]
                mel2ph = sample['clean_mel2ph']  # [B, T_s]
                mel2word = sample['clean_mel2word']
                clean_f0 = sample['clean_f0']  # [B, T_s]
                clean_uv = sample['clean_uv']  # [B, T_s] 0/1
                clean_wav_fn = sample['clean_wav_fn']
                spk_id = sample['spk_id']
                emotion_id = sample['emotion_id']
                emo_embedding = sample['emo_embedding']
                # with gt duration,clean embedding
                # (txt_tokens,spk_id=spk_id, emotion_id=emotion_id,mel2ph=mel2ph, ref_mel2ph=mel2ph, ref_mel2word=mel2word,
                #                        clean_spk_embed=clean_spk_embed, noisy_spk_embed=clean_spk_embed,
                #                        clean_emo_embed=clean_emo_embed, noisy_emo_embed=clean_emo_embed,
                #                        ref_mels=target, f0=clean_f0, uv=clean_uv, tgt_mels=target, global_steps=self.global_step, infer=False)
                # model_out1 = self.model(txt_tokens, spk_id=spk_id, emotion_id=emotion_id, mel2ph=clean_mel2ph, ref_mel2ph=clean_mel2ph, ref_mel2word=clean_mel2word,
                #                         clean_spk_embed=clean_spk_embed, noisy_spk_embed=clean_spk_embed,
                #                         clean_emo_embed=clean_emo_embed, noisy_emo_embed=clean_emo_embed,
                #                        ref_mels=clean_ref_mels, global_steps=self.global_step, infer=True)
                model_out1 = self.model(txt_tokens=txt_tokens,text=text, spk_id=spk_id, emotion_id=emotion_id, mel2ph=mel2ph, ref_mel2ph=mel2ph,
                               ref_mel2word=mel2word,f0=clean_f0, uv=clean_uv,emo_embedding=emo_embedding,
                               clean_wav_fn=clean_wav_fn,spk_ref_mel=target,
                               emo_ref_mel=target,
                                        global_steps=self.global_step, infer=True)
                wav_pred1 = self.vocoder.spec2wav(model_out1['mel_out'][0])
                self.logger.add_audio(f'wav_gtdur_cleanEmbed_{batch_idx}', wav_pred1, self.global_step,
                                      hparams['audio_sample_rate'])
        return outputs

    ############
    # infer
    ############
    def test_step(self, sample, batch_idx):
        text = sample['text']
        txt_tokens = sample['txt_tokens']
        spk_id = sample.get('spk_id')
        emotion_id = sample.get('emotion_id')
        mel2ph, uv, f0 = None, None, None
        ref_mel2word = sample['clean_mel2word']
        ref_mel2ph = sample['clean_mel2ph']
        ref_mels = sample['clean_mels']
        spk_ref_mel = sample['clean_mels']
        emo_ref_mel = sample['clean_mels']
        clean_wav_fn = sample['clean_wav_fn']
        emo_embedding = sample['emo_embedding']
        text = sample['text']
        if hparams['use_gt_dur']:
            mel2ph = sample['clean_mel2ph']
        if hparams['use_gt_f0']:
            f0 = sample['clean_f0']
            uv = sample['clean_uv']
        global_steps = 200000
        run_model = lambda: self.model(
            txt_tokens=txt_tokens, spk_id=spk_id, emotion_id=emotion_id,clean_wav_fn=clean_wav_fn,text = text,
            spk_ref_mel=spk_ref_mel, emo_ref_mel=ref_mels,
            mel2ph=mel2ph, ref_mel2ph=ref_mel2ph, ref_mel2word=ref_mel2word,emo_embedding=emo_embedding,
            f0=f0, uv=uv, ref_mels=ref_mels, global_steps=global_steps, infer=True, ckeck=False
        )
        outputs = run_model()
        sample['outputs'] = self.model.out2mel(outputs['mel_out'])
        sample['mel2ph_pred'] = outputs.get('clean_mel2ph')
        if hparams['use_pitch_embed']:
            sample['clean_f0'] = clean_denorm_f0(sample['clean_f0'], sample['clean_uv'], hparams)
            if hparams['pitch_type'] == 'ph':
                sample['clean_f0'] = torch.gather(F.pad(sample['clean_f0'], [1, 0]), 1, sample['clean_mel2ph'])
            sample['clean_f0_pred'] = outputs.get('clean_f0_denorm')

        return self.after_infer(sample)

    def after_infer(self, predictions, sil_start_frame=0):
        predictions = utils.unpack_dict_to_list(predictions)
        assert len(predictions) == 1, 'Only support batch_size=1 in inference.'
        prediction = predictions[0]
        prediction = utils.tensors_to_np(prediction)
        item_name = prediction.get('item_name')
        text = prediction.get('text')
        ph_tokens = prediction.get('txt_tokens')
        mel_gt = prediction["clean_mels"]
        mel2ph_gt = prediction.get("mel2ph")
        mel2ph_gt = mel2ph_gt if mel2ph_gt is not None else None
        mel_pred = prediction["outputs"]
        mel2ph_pred = prediction.get("mel2ph_pred")
        f0_gt = prediction.get("f0")
        f0_pred = prediction.get("f0_pred")

        str_phs = None
        if self.phone_encoder is not None and 'txt_tokens' in prediction:
            str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)

        if 'encdec_attn' in prediction:
            encdec_attn = prediction['encdec_attn']  # (1, Tph, Tmel)
            encdec_attn = encdec_attn[encdec_attn.max(-1).sum(-1).argmax(-1)]
            txt_lengths = prediction.get('txt_lengths')
            encdec_attn = encdec_attn.T[:, :txt_lengths]
        else:
            encdec_attn = None

        mel_pred_tensor = torch.tensor(mel_pred, dtype=torch.float32)
        wav_pred = self.vocoder.spec2wav(mel_pred_tensor, f0=f0_pred)
        wav_pred[:sil_start_frame * hparams['hop_size']] = 0
        gen_dir = self.gen_dir
        base_fn = f'[{self.results_id:06d}][{item_name}][%s]'
        base_fn = base_fn.replace(' ', '_')
        if not hparams['profile_infer']:
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            if hparams.get('save_mel_npy', False):
                os.makedirs(f'{gen_dir}/npy', exist_ok=True)
            if 'encdec_attn' in prediction:
                os.makedirs(f'{gen_dir}/attn_plot', exist_ok=True)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, mel_pred, base_fn % 'tts', gen_dir, str_phs, mel2ph_pred, encdec_attn]))

            if mel_gt is not None and hparams['save_gt']:
                wav_gt_tensor = torch.tensor(mel_gt, dtype=torch.float32)
                wav_gt = self.vocoder.spec2wav(wav_gt_tensor, f0=f0_gt)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, mel_gt, base_fn % 'Ref', gen_dir, str_phs, mel2ph_gt]))
                if hparams['save_f0']:
                    f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                    f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                    fig = plt.figure()
                    plt.plot(f0_pred_, label=r'$\hat{f_0}$')
                    plt.plot(f0_gt_, label=r'$f_0$')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                    plt.close(fig)

        self.results_id += 1
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.phone_encoder.decode(ph_tokens.tolist()),
            'wav_fn_pred': base_fn % 'tts',
            'wav_fn_gt': base_fn % 'Ref',
        }

    @staticmethod
    def save_result(wav_out, mel, base_fn, gen_dir, str_phs=None, mel2ph=None, alignment=None):
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        f0, _ = get_pitch(wav_out, mel, hparams)
        f0 = f0 / 10 * (f0 > 0)
        plt.plot(f0, c='white', linewidth=1, alpha=0.6)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs.split(" ")
            dur = mel2ph_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                shift = (i % 20) + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black',
                           alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png')
        plt.close(fig)
        if hparams.get('save_mel_npy', False):
            np.save(f'{gen_dir}/npy/{base_fn}', mel)
        if alignment is not None:
            fig, ax = plt.subplots(figsize=(12, 16))
            im = ax.imshow(alignment, aspect='auto', origin='lower',
                           interpolation='none')
            ax.set_xticks(np.arange(0, alignment.shape[1], 5))
            ax.set_yticks(np.arange(0, alignment.shape[0], 10))
            ax.set_ylabel("$S_p$ index")
            ax.set_xlabel("$H_c$ index")
            fig.colorbar(im, ax=ax)
            fig.savefig(f'{gen_dir}/attn_plot/{base_fn}_attn.png', format='png')
            plt.close(fig)
