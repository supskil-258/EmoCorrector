import numpy as np
from modules.EmoCorrector.model.glow_modules import Glow
from modules.fastspeech.tts_modules import PitchPredictor
from modules.EmoCorrector.model.prosody_util import ProsodyAligner, LocalStyleAdaptor
from utils.pitch_utils import f0_to_coarse, clean_denorm_f0
from modules.commons.common_layers import *
import torch.distributions as dist
from utils.hparams import hparams
from modules.EmoCorrector.model.mixstyle import MixStyle
from modules.fastspeech.fs2 import FastSpeech2
from modules.fastspeech.tts_modules import DEFAULT_MAX_SOURCE_POSITIONS
from modules.EmoCorrector.model.glow_modules import LayerNorm as LayerNorm1
from StyleSpeech.models.Modules import Mish
from StyleSpeech.models.Modules import MultiHeadAttention
from StyleSpeech.models.Modules import Conv1dGLU
from torch.autograd import Variable
from torch.autograd import Function
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def init_weights_func(m):
	classname = m.__class__.__name__
	if classname.find("Conv1d") != -1:
		torch.nn.init.xavier_uniform_(m.weight)

def linear(inp, layer):
	batch_size = inp.size(0)
	hidden_dim = inp.size(1)
	seg_len = inp.size(2)
	inp_permuted = inp.permute(0, 2, 1)
	inp_expand = inp_permuted.contiguous().view(batch_size*seg_len, hidden_dim)
	out_expand = layer(inp_expand)
	out_permuted = out_expand.view(batch_size, seg_len, out_expand.size(1))
	out = out_permuted.permute(0, 2, 1)
	return out

def gumbel_softmax(logits, temperature=0.1):
	def _sample_gumbel(shape, eps=1e-20):
		U = torch.rand(shape, requires_grad=True)
		dist = -Variable(torch.log(-torch.log(U + eps) + eps))
		return dist.cuda() if torch.cuda.is_available() else dist
	def _gumbel_softmax_sample(logits, temperature):
		y = logits + _sample_gumbel(logits.size())
		return F.softmax(y / temperature, dim=-1)

	y = _gumbel_softmax_sample(logits, temperature)
	shape = y.size()
	_, ind = y.max(dim=-1)
	y_hard = torch.zeros_like(y).view(-1, shape[-1])
	y_hard.scatter_(1, ind.view(-1, 1), 1)
	y_hard = y_hard.view(*shape)
	return (y_hard - y).detach() + y

def temporal_avg_pool(x, mask=None):
	if mask is None:
		out = torch.mean(x, dim=1)
	else:
		len_ = (~mask).sum(dim=1).unsqueeze(1)
		x = x.masked_fill(mask.unsqueeze(-1), 0)
		x = x.sum(dim=1)
		out = torch.div(x, len_)
	return out
class ReferenceEncoder(nn.Module):
	def __init__(self,n_mel_channels):
		super(ReferenceEncoder, self).__init__()
		E = 256
		n_mel_channels = n_mel_channels
		ref_enc_filters = [32, 32, 64, 64, 128, 128]
		ref_enc_size = [3, 3]
		ref_enc_strides = [2, 2]
		ref_enc_pad = [1, 1]
		self.n_mel_channels = n_mel_channels
		K = len(ref_enc_filters)
		filters = [1] + ref_enc_filters
		convs = [nn.Conv2d(in_channels=filters[i],
						   out_channels=filters[i + 1],
						   kernel_size=ref_enc_size,
						   stride=ref_enc_strides,
						   padding=ref_enc_pad) for i in range(K)]
		self.convs = nn.ModuleList(convs)
		self.bns = nn.ModuleList(
			[nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)])
		out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
		self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
						  hidden_size=E,
						  batch_first=True)
		self.fc = nn.Linear(256,256)


	def forward(self, inputs):
		N = inputs.size(0)
		out = inputs.view(N, 1, -1, self.n_mel_channels)
		for conv, bn in zip(self.convs, self.bns):
			out = conv(out)
			out = bn(out)
			out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]
		out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
		T = out.size(1)
		N = out.size(0)
		out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

		self.gru.flatten_parameters()
		memory, out = self.gru(out)  # out --- [1, N, E//2]
		out = out.squeeze(0)
		out = self.fc(out)
		return out
	def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
		for i in range(n_convs):
			L = (L - kernel_size + 2 * pad) // stride + 1
		return L
class GRL2(Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.view_as(x)
	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha
		return output,None
class Ortho_Loss(nn.Module):
	def __init__(self):
		super(Ortho_Loss,self).__init__()
	def forward(self,x,y):
		plotValue = torch.mul(x,y)
		normValue = torch.norm(plotValue,p=2)
		powerNorm = torch.pow(normValue,2)
		return powerNorm/x.shape[0]

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)
class ResidualBlock(nn.Module):
	def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0,
				 c_multiple=2, ln_eps=1e-12):
		super(ResidualBlock, self).__init__()

		if norm_type == 'bn':
			norm_builder = lambda: nn.BatchNorm1d(channels)
		elif norm_type == 'in':
			norm_builder = lambda: nn.InstanceNorm1d(channels, affine=True)
		elif norm_type == 'gn':
			norm_builder = lambda: nn.GroupNorm(8, channels)
		elif norm_type == 'ln':
			norm_builder = lambda: LayerNorm1(channels,  eps=ln_eps)
		else:
			norm_builder = lambda: nn.Identity()

		self.blocks = [
			nn.Sequential(
				norm_builder(),
				nn.Conv1d(channels, c_multiple * channels, kernel_size, dilation=dilation,
						  padding=(dilation * (kernel_size - 1)) // 2),
				LambdaLayer(lambda x: x * kernel_size ** -0.5),
				nn.GELU(),
				nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation),
			)
			for i in range(n)
		]
		self.blocks = nn.ModuleList(self.blocks)
		self.dropout = dropout

	def forward(self, x):
		nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
		for b in self.blocks:
			x_ = b(x)
			if self.dropout > 0 and self.training:
				x_ = F.dropout(x_, self.dropout, training=self.training)
			x = x + x_
			x = x * nonpadding
		return x
class ConvBlocks(nn.Module):
	def __init__(self, channels, out_dims, dilations, kernel_size,
				 norm_type='ln', layers_in_block=2, c_multiple=2,
				 dropout=0.0, ln_eps=1e-5, init_weights=True):
		super(ConvBlocks, self).__init__()
		self.res_blocks = nn.Sequential(
			*[ResidualBlock(channels, kernel_size, d,
							n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple,
							dropout=dropout, ln_eps=ln_eps)
			  for d in dilations],
		)
		if norm_type == 'bn':
			norm = nn.BatchNorm1d(channels)
		elif norm_type == 'in':
			norm = nn.InstanceNorm1d(channels, affine=True)
		elif norm_type == 'gn':
			norm = nn.GroupNorm(8, channels)
		elif norm_type == 'ln':
			norm = LayerNorm1(channels, eps=ln_eps)

		self.last_norm = norm
		self.post_net1 = nn.Conv1d(channels, out_dims, kernel_size=3, padding=1)
		if init_weights:
			self.apply(init_weights_func)

	def forward(self, x):
		"""

		:param x: [B, T, H]
		:return:  [B, T, H]
		"""
		x = x.transpose(1, 2)
		nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
		x = self.res_blocks(x) * nonpadding
		x = self.last_norm(x) * nonpadding
		x = self.post_net1(x) * nonpadding
		return x.transpose(1, 2)

class MelStyleEncoder256(nn.Module):
	''' MelStyleEncoder '''

	def __init__(self):
		super(MelStyleEncoder256, self).__init__()
		self.in_dim = 80
		self.hidden_dim = 256
		self.out_dim = 256
		self.kernel_size = 5
		self.n_head = 2
		self.dropout = 0.4

		self.spectral = nn.Sequential(
			LinearNorm(self.in_dim, self.hidden_dim),
			Mish(),
			nn.Dropout(self.dropout),
			LinearNorm(self.hidden_dim, self.hidden_dim),
			Mish(),
			nn.Dropout(self.dropout)
		)

		self.temporal = nn.Sequential(
			Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
			Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
		)

		self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim,
										   self.hidden_dim // self.n_head, self.hidden_dim // self.n_head, self.dropout)

		self.fc = LinearNorm(self.hidden_dim, self.out_dim)

	def temporal_avg_pool(self, x, mask=None):
		if mask is None:
			out = torch.mean(x, dim=1)
		else:
			len_ = (~mask).sum(dim=1).unsqueeze(1)
			x = x.masked_fill(mask.unsqueeze(-1), 0)
			x = x.sum(dim=1)
			out = torch.div(x, len_)
		return out

	def forward(self, x, mask=None):
		max_len = x.shape[1]
		slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None

		# spectral
		x = self.spectral(x)
		# temporal
		x = x.transpose(1, 2)
		x = self.temporal(x)
		x = x.transpose(1, 2)
		checkx = x
		# self-attention
		if mask is not None:
			x = x.masked_fill(mask.unsqueeze(-1), 0)
			checkeq = torch.eq(checkx,x)
		x, _ = self.slf_attn(x, mask=slf_attn_mask)
		# fc
		x = self.fc(x)
		# temoral average pooling
		w = self.temporal_avg_pool(x, mask=mask).unsqueeze(1)

		return x,w
class MiMelStyleEncoder(nn.Module):
	''' MelStyleEncoder '''

	def __init__(self):
		super(MiMelStyleEncoder, self).__init__()
		self.in_dim = 256
		self.hidden_dim = 256
		self.out_dim = 256
		self.kernel_size = 5
		self.n_head = 2
		self.dropout = 0.4
		self.spectral = nn.Sequential(
			LinearNorm(self.in_dim, self.hidden_dim),
			Mish(),
			nn.Dropout(self.dropout),
			LinearNorm(self.hidden_dim, self.hidden_dim),
			Mish(),
			nn.Dropout(self.dropout)
		)
		self.temporal = nn.Sequential(
			Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
			Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
		)
		self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim,
										   self.hidden_dim // self.n_head, self.hidden_dim // self.n_head, self.dropout)
		self.fc = LinearNorm(self.hidden_dim, self.out_dim)

	def temporal_avg_pool(self, x, mask=None):
		if mask is None:
			out = torch.mean(x, dim=1)
		else:
			len_ = (~mask).sum(dim=1).unsqueeze(1)
			x = x.masked_fill(mask.unsqueeze(-1), 0)
			x = x.sum(dim=1)
			out = torch.div(x, len_)
		return out

	def forward(self, x, mask=None):
		max_len = x.shape[1]
		slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
		# spectral
		x = self.spectral(x)
		# temporal
		x = x.transpose(1, 2)
		x = self.temporal(x)
		x = x.transpose(1, 2)
		checkx = x
		# self-attention
		if mask is not None:
			x = x.masked_fill(mask.unsqueeze(-1), 0)
			checkeq = torch.eq(checkx,x)
		x, _ = self.slf_attn(x, mask=slf_attn_mask)
		# fc
		x = self.fc(x)
		# temoral average pooling
		w = self.temporal_avg_pool(x, mask=mask).unsqueeze(1)

		return x,w
class CLAPTextEncoder():
	def __init__(self, model_path="retrieval/RoBERTa", gpu_id=0):
		super(CLAPTextEncoder, self).__init__()
		self.model_path = model_path
		self.gpu_id = gpu_id
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
		for param in self.model.parameters():
			param.requires_grad = True
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
		print(f"CLAPTextEncoder Model loaded on {self.device}")


	def forward(self, text):
		inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
		inputs = {key: value.to(self.device) for key, value in inputs.items()}
		outputs = self.model(**inputs, output_hidden_states=True)
		cls_hidden_state = outputs.hidden_states[-1][:, 0, :]
		return cls_hidden_state

class Adapter(nn.Module):
	def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024,dropout_prob=0.3):
		super(Adapter, self).__init__()
		self.layer1 = nn.Linear(input_dim, hidden_dim)
		self.bn1 = nn.BatchNorm1d(hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, hidden_dim)
		self.bn2 = nn.BatchNorm1d(hidden_dim)
		self.layer3 = nn.Linear(hidden_dim, output_dim)
		self.bn3 = nn.BatchNorm1d(output_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout_prob)
		self.residual = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		residual = self.residual(x)
		out = self.relu(self.bn1(self.layer1(x)))
		out = self.dropout(out)
		out = self.relu(self.bn2(self.layer2(out)))
		out = self.dropout(out)
		out = self.bn3(self.layer3(out))
		out += residual
		out = self.relu(out)
		return out
class MultiModalProjection(nn.Module):
	def __init__(self, emo_embed_dim, text_embed_dim, joint_space_dim, hidden_dim):
		super(MultiModalProjection, self).__init__()
		self.emo_proj = nn.Sequential(
			nn.Linear(emo_embed_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, joint_space_dim)
		)
		self.text_proj = nn.Sequential(
			nn.Linear(text_embed_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, joint_space_dim)
		)

	def forward(self, global_emo_embed=None, text_embed=None):
		if text_embed is not None:
			projected_text = self.text_proj(text_embed)
		else:
			projected_text = None

		if global_emo_embed is not None:
			projected_emo = self.emo_proj(global_emo_embed)
		else:
			projected_emo = None

		return projected_emo, projected_text

class EmoCorrector(FastSpeech2):

	def __init__(self, dictionary, out_dims=None):
		super().__init__(dictionary, out_dims)
		self.norm = MixStyle(p=0.5, alpha=0.1, eps=1e-6, hidden_size=self.hidden_size)
		self.ce_loss = F.cross_entropy
		self.op_lambda = 1
		self.GRL_alpha = 1.
		self.GRL2 = GRL2()
		self.emo_emo_classifer = nn.Sequential(nn.Linear(1024, 5),nn.Softmax(dim=1))#768
		num_speaker = 12
		self.spk_spk_classifer = nn.Sequential(nn.Linear(256, num_speaker),nn.Softmax(dim=1))
		self.spk_emo_classifer = nn.Sequential(nn.Linear(256, 5),nn.Softmax(dim=1))
		self.emo_spk_classifer = nn.Sequential(nn.Linear(1024, num_speaker),nn.Softmax(dim=1))
		self.emo_fc = nn.Linear(1024, 256)#768
		self.CLAPtext_encoder = CLAPTextEncoder()
		self.audio_adaptor = Adapter(input_dim=1024, hidden_dim=1024, output_dim=1024)
		self.MultiModalProjection = MultiModalProjection(emo_embed_dim=1024, text_embed_dim=768, joint_space_dim=1024,hidden_dim=2048)
		self.prosody_extractor_utter = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx)
		self.l1_utter = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.align_utter = ProsodyAligner(num_layers=2)
		## phoneme level
		self.prosody_extractor_ph = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx)
		self.l1_ph = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.align_ph = ProsodyAligner(num_layers=2)
		## word level
		self.prosody_extractor_word = LocalStyleAdaptor(self.hidden_size, hparams['nVQ'], self.padding_idx)
		self.l1_word = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.align_word = ProsodyAligner(num_layers=2)
		self.pitch_inpainter_predictor = PitchPredictor(
			self.hidden_size, n_chans=self.hidden_size,
			n_layers=3, dropout_rate=0.1, odim=2,
			padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])

		# build attention layer
		self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
		self.embed_positions = SinusoidalPositionalEmbedding(
			self.hidden_size, self.padding_idx,
			init_size=self.max_source_positions + self.padding_idx + 1,
		)
		# build post flow
		cond_hs = 80
		if hparams.get('use_txt_cond', True):
			cond_hs = cond_hs + hparams['hidden_size']
		cond_hs = cond_hs + hparams['hidden_size'] * 2
		self.post_flow = Glow(
			80, hparams['post_glow_hidden'], hparams['post_glow_kernel_size'], 1,
			hparams['post_glow_n_blocks'], hparams['post_glow_n_block_layers'],
			n_split=4, n_sqz=2,
			gin_channels=cond_hs,
			share_cond_layers=hparams['post_share_cond_layers'],
			share_wn_layers=hparams['share_wn_layers'],
			sigmoid_scale=hparams['sigmoid_scale']
		)
		self.prior_dist = dist.Normal(0, 1)
		self.reference_encoder_spk = ReferenceEncoder(80)

	def forward(self, text = None, txt_tokens=None, spk_id=None, emotion_id=None, mel2ph=None, ref_mel2ph=None, ref_mel2word=None,
				spk_ref_mel=None, emo_ref_mel=None,clean_wav_fn=None,emo_embedding=None,
				f0=None, uv=None, skip_decoder=False, global_steps=0,infer=False,check=False,
				**kwargs):

		ret = {}
		"""
		加入了文本-音频情感检索

		"""
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		global_emo_embed = self.audio_adaptor(emo_embedding).to(device)
		text_embed = self.CLAPtext_encoder.forward(text)
		if global_steps <200000 :
			global_emo_embed_emo = emo_embedding.squeeze(1)
			projected_audio, projected_text = self.MultiModalProjection(global_emo_embed_emo, text_embed)
			global_emo_embed_emo = F.normalize(projected_audio, p=2, dim=1)
			text_embed = F.normalize(projected_text, p=2, dim=1)

			for i in range(len(clean_wav_fn)):
				clean_wav_basename = os.path.splitext(os.path.basename(clean_wav_fn[i]))[0]
				projected_audio_layer = global_emo_embed_emo[i]
				projected_text_layer = text_embed[i]
				# define embedding path
				audio_save_path = os.path.join('data/emo_clap/audio',
											   f'{clean_wav_basename}.npy')
				text_save_path = os.path.join('data/emo_clap/text',
											  f'{clean_wav_basename}.npy')
				os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)
				os.makedirs(os.path.dirname(text_save_path), exist_ok=True)

				np.save(audio_save_path, projected_audio_layer.unsqueeze(0).cpu().detach().numpy())
				np.save(text_save_path, projected_text_layer.unsqueeze(0).cpu().detach().numpy())
				ret["projected_audio"] = projected_audio_layer.unsqueeze(0).cpu().detach().numpy()
				ret["projected_text"] = projected_text_layer.unsqueeze(0).cpu().detach().numpy()
			# make sure the same size
			assert global_emo_embed_emo.shape == text_embed.shape, "Shape mismatch between embeddings!"
			# matrix shape [batch_size, batch_size]
			temperature = 0.1
			similarity_matrix = torch.matmul(global_emo_embed_emo, text_embed.T) / temperature
			# Construct labels: positive sample pairs are on the diagonal
			# Assumes positive sample pairs are in the same batch, where each audio corresponds to the text in the same row
			batch_size = similarity_matrix.size(0)
			assert similarity_matrix.shape == (batch_size,
											   batch_size), f"Expected similarity_matrix shape {[batch_size, batch_size]}, got {similarity_matrix.shape}"
			labels = torch.arange(batch_size).to(similarity_matrix.device)
			if batch_size == 1:
				fake_text_embed = torch.ones_like(text_embed)
				fake_emo_embed_emo = torch.ones_like(global_emo_embed_emo)
				text_embed = torch.cat((text_embed, fake_text_embed), dim=0)
				global_emo_embed_emo = torch.cat((global_emo_embed_emo, fake_emo_embed_emo), dim=0)
				similarity_matrix = torch.matmul(global_emo_embed_emo, text_embed.T) / temperature
				labels = torch.cat([torch.arange(batch_size).to(device), torch.ones(1).to(device)], dim=0).to(
					torch.long)

			loss_CLAP = F.cross_entropy(similarity_matrix, labels)
			ret['loss_CLAP'] = loss_CLAP

#####################################################################################################
		encoder_out = self.encoder(txt_tokens)  # [B, T, C]
		src_nonpadding = (txt_tokens > 0).float()[:, :, None]
		global_spk_embed = self.reference_encoder_spk(spk_ref_mel).unsqueeze(1)
		emotion_id = torch.Tensor(emotion_id).long().cuda()
		spk_id = torch.Tensor(spk_id).long().cuda()

		pred_emo2emo = self.emo_emo_classifer(global_emo_embed.squeeze(1))
		clean_ce_loss_emo2emo = self.ce_loss(pred_emo2emo, emotion_id.cuda())
		ret['clean_loss_emo2emo'] = clean_ce_loss_emo2emo

		pred_spk2spk = self.spk_spk_classifer(global_spk_embed.squeeze(1))
		clean_ce_loss_spk2spk = self.ce_loss(pred_spk2spk, spk_id.cuda())
		ret['clean_loss_spk2spk'] = clean_ce_loss_spk2spk


		index_emo2emo = torch.argmax(pred_emo2emo.squeeze(1), dim=1)
		num_emo2emo = 0
		for i in range(0, len(index_emo2emo)):
			if index_emo2emo[i] == emotion_id[i]:
				num_emo2emo = num_emo2emo + 1
		emo2emo_acc = num_emo2emo / len(index_emo2emo)
		ret['clean_emo2emo_acc'] = emo2emo_acc

		index_spk2spk = torch.argmax(pred_spk2spk.squeeze(1), dim=1)
		num_spk2spk = 0
		for i in range(0, len(index_spk2spk)):
			if index_spk2spk[i] == spk_id[i]:
				num_spk2spk = num_spk2spk + 1
		spk2spk_acc = num_spk2spk / len(index_spk2spk)
		ret['clean_spk2spk_acc'] = spk2spk_acc

		if infer:
			dur_inp = (encoder_out + global_spk_embed) * src_nonpadding
			ref_mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)

		grl_global_spk_embed = global_spk_embed
		grl_global_spk_embed = self.GRL2.apply(grl_global_spk_embed, self.GRL_alpha)
		pred_spk2emo = self.spk_emo_classifer(grl_global_spk_embed.squeeze(1))
		clean_grl_loss_spk2emo = self.ce_loss(pred_spk2emo, emotion_id.cuda())
		ret['clean_loss_grl_spk2emo'] = clean_grl_loss_spk2emo

		grl_global_emo_embed = global_emo_embed
		grl_global_emo_embed  = self.GRL2.apply(grl_global_emo_embed , self.GRL_alpha)
		pred_emo2spk = self.emo_spk_classifer(grl_global_emo_embed .squeeze(1))
		clean_grl_loss_emo2spk = self.ce_loss(pred_emo2spk, spk_id.cuda())
		ret['clean_loss_grl_emo2spk'] = clean_grl_loss_emo2spk

		index_spk2emo = torch.argmax(pred_spk2emo.squeeze(1), dim=1)
		num_spk2emo = 0
		for i in range(0, len(index_spk2emo)):
			if index_spk2emo[i] == emotion_id[i]:
				num_spk2emo += 1
		if len(index_spk2emo) > 0:
			spk2emo_acc = num_spk2emo / len(index_spk2emo)
		else:
			spk2emo_acc = 0
		ret['clean_spk2emo_acc'] = spk2emo_acc

		if global_steps>20000:
			pred_emo_embed = self.emo_fc(global_emo_embed.to(device))
			pred_emo_embed = pred_emo_embed.unsqueeze(1)
			pred_spk_embed = global_spk_embed

			dur_inp = (encoder_out + pred_spk_embed + pred_emo_embed) * src_nonpadding
			mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)
			tgt_nonpadding2 = (mel2ph > 0).float()[:, :, None]
			decoder_inp = self.expand_states(encoder_out, mel2ph)
			decoder_inp = self.norm(decoder_inp, pred_emo_embed + pred_spk_embed)
			ret['ref_mel2ph'] = ref_mel2ph
			ret['ref_mel2word'] = ref_mel2word
			pitch_inp_domain_agnostic = decoder_inp * tgt_nonpadding2
			pitch_inp_domain_specific = (
													decoder_inp + pred_spk_embed + pred_emo_embed) * tgt_nonpadding2
			predicted_pitch = self.inpaint_pitch(pitch_inp_domain_agnostic, pitch_inp_domain_specific, f0, uv, mel2ph,
												 ret)

			decoder_inp = decoder_inp + pred_spk_embed + pred_emo_embed + predicted_pitch
			ret['decoder_inp'] = decoder_inp = decoder_inp * tgt_nonpadding2
			if skip_decoder:
				return ret
			ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding2, ret, infer=infer, **kwargs)
			# postflow
			is_training = self.training
			ret['x_mask'] = tgt_nonpadding2
			ret['spk_embed'] = pred_spk_embed
			ret['emo_embed'] = pred_emo_embed
			self.run_post_glow(spk_ref_mel, infer, is_training, ret)
		return ret

	def get_prosody_ph(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
		if global_steps > hparams['vq_start'] or infer:
			prosody_embedding, loss, ppl = self.prosody_extractor_ph(ref_mels, ret['ref_mel2ph'], no_vq=False)
			ret['vq_loss_ph'] = loss
			ret['ppl_ph'] = ppl
		else:
			prosody_embedding = self.prosody_extractor_ph(ref_mels, ret['ref_mel2ph'], no_vq=True)
		positions = self.embed_positions(prosody_embedding[:, :, 0])
		prosody_embedding = self.l1_ph(torch.cat([prosody_embedding, positions], dim=-1))

		src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx).data
		prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx).data
		if global_steps < hparams['forcing']:
			output, guided_loss, attn_emo = self.align_ph(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
												   src_key_padding_mask, prosody_key_padding_mask, forcing=True)
		else:
			output, guided_loss, attn_emo = self.align_ph(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
													   src_key_padding_mask, prosody_key_padding_mask, forcing=False)

		ret['gloss_ph'] = guided_loss
		ret['attn_ph'] = attn_emo
		return output.transpose(0, 1)

	def get_prosody_word(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
		if global_steps > hparams['vq_start'] or infer:
			prosody_embedding, loss, ppl = self.prosody_extractor_word(ref_mels, ret['ref_mel2word'], no_vq=False)
			ret['vq_loss_word'] = loss
			ret['ppl_word'] = ppl
		else:
			prosody_embedding = self.prosody_extractor_word(ref_mels, ret['ref_mel2word'], no_vq=True)

		positions = self.embed_positions(prosody_embedding[:, :, 0])
		prosody_embedding = self.l1_word(torch.cat([prosody_embedding, positions], dim=-1))

		src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx).data
		prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx).data
		if global_steps < hparams['forcing']:
			output, guided_loss, attn_emo = self.align_word(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
												   src_key_padding_mask, prosody_key_padding_mask, forcing=True)
		else:
			output, guided_loss, attn_emo = self.align_word(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
													   src_key_padding_mask, prosody_key_padding_mask, forcing=False)
		ret['gloss_word'] = guided_loss
		ret['attn_word'] = attn_emo
		return output.transpose(0, 1)

	def get_prosody_utter(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
		if global_steps > hparams['vq_start'] or infer:
			prosody_embedding, loss, ppl = self.prosody_extractor_utter(ref_mels, no_vq=False)
			ret['vq_loss_utter'] = loss
			ret['ppl_utter'] = ppl
		else:
			prosody_embedding = self.prosody_extractor_utter(ref_mels, no_vq=True)

		positions = self.embed_positions(prosody_embedding[:, :, 0])
		prosody_embedding = self.l1_utter(torch.cat([prosody_embedding, positions], dim=-1))

		src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx).data
		prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx).data
		if global_steps < hparams['forcing']:
			output, guided_loss, attn_emo = self.align_utter(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
												   src_key_padding_mask, prosody_key_padding_mask, forcing=True)
		else:
			output, guided_loss, attn_emo = self.align_utter(encoder_out.transpose(0, 1), prosody_embedding.transpose(0, 1),
													   src_key_padding_mask, prosody_key_padding_mask, forcing=False)
		ret['gloss_utter'] = guided_loss
		ret['attn_utter'] = attn_emo
		return output.transpose(0, 1)

	def inpaint_pitch(self, pitch_inp_domain_agnostic, pitch_inp_domain_specific, f0, uv, mel2ph, ret):
		if hparams['pitch_type'] == 'frame':
			pitch_padding = mel2ph == 0
		if hparams['predictor_grad'] != 1:
			pitch_inp_domain_agnostic = pitch_inp_domain_agnostic.detach() + hparams['predictor_grad'] * (pitch_inp_domain_agnostic - pitch_inp_domain_agnostic.detach())
			pitch_inp_domain_specific = pitch_inp_domain_specific.detach() + hparams['predictor_grad'] * (pitch_inp_domain_specific - pitch_inp_domain_specific.detach())

		pitch_domain_agnostic = self.pitch_predictor(pitch_inp_domain_agnostic)
		pitch_domain_specific = self.pitch_inpainter_predictor(pitch_inp_domain_specific)
		pitch_pred = pitch_domain_agnostic + pitch_domain_specific
		ret['pitch_pred'] = pitch_pred

		use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']
		if f0 is None:
			f0 = pitch_pred[:, :, 0]  # [B, T]
			if use_uv:
				uv = pitch_pred[:, :, 1] > 0  # [B, T]
		clean_f0_denorm = clean_denorm_f0(f0, uv if use_uv else None, hparams, pitch_padding=pitch_padding)
		clean_pitch = f0_to_coarse(clean_f0_denorm)  # start from 0 [B, T_txt]
		ret['clean_f0_denorm'] = clean_f0_denorm
		ret['clean_f0_denorm_pred'] = clean_denorm_f0(pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None, hparams, pitch_padding=pitch_padding)
		if hparams['pitch_type'] == 'ph':
			clean_pitch = torch.gather(F.pad(clean_pitch, [1, 0]), 1, mel2ph)
			ret['clean_f0_denorm'] = torch.gather(F.pad(ret['clean_f0_denorm'], [1, 0]), 1, mel2ph)
			ret['clean_f0_denorm_pred'] = torch.gather(F.pad(ret['clean_f0_denorm_pred'], [1, 0]), 1, mel2ph)
		pitch_embed = self.pitch_embed(clean_pitch)
		return pitch_embed

	def run_post_glow(self, tgt_mels, infer, is_training, ret):
		x_recon = ret['mel_out'].transpose(1, 2)
		g = x_recon
		B, _, T = g.shape
		if hparams.get('use_txt_cond', True):
			g = torch.cat([g, ret['decoder_inp'].transpose(1, 2)], 1)
		g_spk_embed = ret['spk_embed'].repeat(1, T, 1).transpose(1, 2)
		g_emo_embed = ret['emo_embed'].repeat(1, T, 1).transpose(1, 2)
		g = torch.cat([g, g_spk_embed, g_emo_embed], dim=1)
		prior_dist = self.prior_dist
		if not infer:
			if is_training:
				self.train()
			x_mask = ret['x_mask'].transpose(1, 2)
			y_lengths = x_mask.sum(-1)
			g = g.detach()
			tgt_mels = tgt_mels.transpose(1, 2)
			z_postflow, ldj = self.post_flow(tgt_mels, x_mask, g=g)
			ldj = ldj / y_lengths / 80
			ret['z_pf'], ret['ldj_pf'] = z_postflow, ldj
			ret['postflow'] = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
		else:
			x_mask = torch.ones_like(x_recon[:, :1, :])
			z_post = prior_dist.sample(x_recon.shape).to(g.device) * hparams['noise_scale']
			x_recon_, _ = self.post_flow(z_post, x_mask, g, reverse=True)
			x_recon = x_recon_
			ret['mel_out'] = x_recon.transpose(1, 2)

	def pad_or_cut2(self, noisy_out, dim):
		n_dim = noisy_out.shape[1]
		if n_dim == dim:
			return noisy_out
		elif n_dim > dim:
			return noisy_out[:,:dim,:]
		else:
			noisy_out = F.interpolate(noisy_out.transpose(1, 2), dim).transpose(1, 2)
		return noisy_out


