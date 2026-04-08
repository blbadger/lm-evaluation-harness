import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin, GenerationConfig
import datasets
from datasets import load_from_disk
from safetensors.torch import load_model, save_model
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_model
from datasets import load_dataset, load_from_disk
from accelerate.utils import DistributedDataParallelKwargs
import os
import re
from dotenv import load_dotenv
import shutil
from transformers import TextStreamer
from lm_eval.models.dual_srm_model import DualMixer
import warnings
import time
import uuid
import pprint
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=UserWarning)


class DualMLPMixer(DualMixer, GenerationMixin):

	def __init__(
		self,
		vocab_size: int,
		hidden_dim: int,
		seq_len: int,
		num_blocks: int,
		heads=None,
		kernel=1,
		expanded_convs=False,
		copy=False,        
		mixed_heads=False,
		combined_heads=False,
		decay=False,
		parallel_heads=False,
		use_projections=True,
		dropout_layer=False,
		is_reward_model=False,
		**kwargs
	):
		super().__init__(vocab_size, hidden_dim, seq_len, num_blocks, heads=heads, kernel=kernel, expanded_convs=expanded_convs,
			mixed_heads=mixed_heads, combined_heads=combined_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections)
		self._init_weights()
		self.generation_config = GenerationConfig()
		config  = {
				 'hidden_size':hidden_dim,
				 'intermediate_size': 4*hidden_dim,
				 'num_hidden_layers': num_blocks,
				 'num_attention_heads': 4,
				 'vocab_size': vocab_size
				}
		self.config = LlamaConfig(**config)
		self.hidden_dim = hidden_dim
		self.n_heads = heads
		self.seq_len = seq_len
		self.main_input_name = 'input_ids'
		self._supports_cache_class = False
		self.cache_built = False
		self.device = self.output_layer.weight.device
		self.warnings_issued={}
		self.is_reward_model = is_reward_model
		if is_reward_model:
			self.loss_fn = nn.MSELoss()
			self.reward_head = nn.Linear(self.hidden_dim, 1)
	
	def add_model_tags(self, tag):
		print (tag)

	def gradient_checkpointing_enable(self, *args, **kwargs):
		pass

	def can_generate(self):
		return True

	def count_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def _is_stateful(self):
		return False

	def get_cache(self):
		cache_store = {}
		for block in self.mixer_blocks:
			for h in range(len(block.token_mixing_layer.mixer_heads)):
				cache_store[f'{block},{h}'] = block.token_mixing_layer.mixer_heads[h].cache
		return cache_store

	def load_cache(self, cache_store):
		for block in self.mixer_blocks:
			for h in range(len(block.token_mixing_layer.mixer_heads)):
				block.token_mixing_layer.mixer_heads[h].cache = cache_store[f'{block},{h}']
		return

	def build_cache(self, input_ids):
		for i in range(len(input_ids[0])-1):
			x = self.input_layer(input_ids[:, i])
			for block in self.mixer_blocks:
				x = block(x, i, True)
		self.cache_built = True
		return

	def clear_cache(self):
		for block in self.mixer_blocks:
			for h in range(len(block.token_mixing_layer.mixer_heads)):
				block.token_mixing_layer.mixer_heads[h].cache = torch.zeros(self.hidden_dim//self.n_heads).to('cuda') # only for mixed heads
		self.cache_built = False

	def select_and_expand_cache(self, top_indices, expansion_factor):
		for block in self.mixer_blocks:
			for h in range(len(block.token_mixing_layer.mixer_heads)):
				block.token_mixing_layer.mixer_heads[h].cache = block.token_mixing_layer.mixer_heads[h].cache[top_indices, :].repeat(expansion_factor, 1)

	def forward(self, input_ids, labels=None, is_recurrent=False, **kwargs):
		if not is_recurrent:
			is_recurrent = input_ids.shape[1] < self.seq_len
		print (f'is recurrent: {is_recurrent}')
		# mask pad tokens in labels for loss computation
		if labels is not None:
			labels = torch.where(labels==tokenizer.pad_token_id, -100., labels).to(self.input_layer.weight.dtype)
		if not self.cache_built and is_recurrent:
			self.build_cache(input_ids)
		index = input_ids.shape[1] - 1
		if is_recurrent:
			input_ids = input_ids[:, -1] # last token only

		# model's forward pass
		x = self.input_layer(input_ids)
		for block in self.mixer_blocks:
			x = block(x, index, is_recurrent)

		if not self.is_reward_model:
			logits = self.output_layer(x).unsqueeze(1)
		else:
			# reward model output
			output = self.reward_head(x).squeeze(-1)
			if labels is not None:
				loss = self.loss_fn(output, labels)
			else:
				loss = 0
			return CausalLMOutput(loss=loss, logits=output)

		# policy model output
		if labels is not None:
			shift_logits = logits[:, :-1].contiguous()
			shift_labels = labels[:, 1:].contiguous()
			shift_logits = shift_logits.view(-1, self.vocab_size)
			shift_labels = shift_labels.view(-1)
			loss = self.loss_fn(shift_logits, shift_labels)
			return CausalLMOutput(loss=loss, logits=logits)
		else:
			return CausalLMOutput(loss=0, logits=logits)
