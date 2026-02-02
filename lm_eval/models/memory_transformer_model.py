import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.generation import GenerationMixin, GenerationConfig
from peft.peft_model import PeftModelForCausalLM
import mlflow
from datasets import load_dataset
from mixer_autoencoder import MixerBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def copy_dataset(inputs, blank_copy=False, clone=True):
        if clone:
                input_ids = torch.clone(inputs) # to avoid in-place modification
        else:
                input_ids = inputs
        n_ctx = len(input_ids[0])
        for i, input in enumerate(input_ids):
                first_half = input[:n_ctx//2]
                if blank_copy:
                        copied_halves = torch.cat((first_half, torch.ones(first_half.shape).to(first_half.device))).to(torch.long)
                else:
                        copied_halves = torch.cat((first_half, first_half)).to(torch.long)
                input_ids[i] = copied_halves
        return input_ids

def copy_labels(label_arr, clone=True):
	if clone:
		labels = torch.clone(label_arr) # to avoid in-place modification
	else:
		labels = label_arr
	n_ctx = len(labels[0])
	for i, input in enumerate(labels):
		first_half = input[:n_ctx//2]
		pad_half = torch.ones(first_half.shape).to(device) * -100
		halves = torch.cat((pad_half, first_half)).to(torch.long)
		labels[i] = halves
	return labels
	
class RecurrentMemoryTransformer(nn.Module, GenerationMixin):

	def __init__(self, n_vocab, dim, depth, length, n_heads=4, n_chunks=4):
		super().__init__()

		llama_config_kwargs = {
			'hidden_size': dim,
			'intermediate_size': 4*dim,
			'num_hidden_layers': depth,
			'num_attention_heads': n_heads,
			'vocab_size': n_vocab
		}
		decoder_configuration = LlamaConfig(**llama_config_kwargs)
		self.decoder = LlamaModel(decoder_configuration)
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.chunks = n_chunks
		self.decoder_dim = dim

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
		input_ids = input_ids.to(device)
		total_loss = 0

		for c in range(self.chunks):
			x = input_ids[:, c*self.tokenized_length: (c+1)*self.tokenized_length]
			decoder_embeds = self.decoder_wte(x)
			# attention mask is of shape [b, t]
			if c == 0:
				encoder_embedding = torch.ones((input_ids.shape[0], 1, self.decoder_dim)).to(device)
				attention_insert = torch.zeros(attention_mask.shape[0], 1).to(device)
			else:
				attention_inset = torch.ones(attention_mask.shape[0], 1).to(device)
			
			attention_mask = torch.cat((attention_insert, attention_mask), dim=1)	
			decoder_embeds[:, -1, :] = encoder_embedding.squeeze(1)
			x = torch.cat((encoder_embedding, decoder_embeds), dim=1)
			x = self.decoder(inputs_embeds=x).last_hidden_state #, attention_mask=attention_mask).last_hidden_state
			encoder_embedding = x[:, -1, :].unsqueeze(1)
			output = self.lm_head(x)
			if labels.dim() > 2:
				labels = rearrange(labels, 'b p t -> b (p t)')
			output = rearrange(output, 'b t e -> b e t')
			shift_labels, shift_logits = labels, output
			shift_logits = output[..., 1:-1].contiguous() # first c 'tokens' are encoding
			shift_labels = labels[..., (c*self.tokenized_length)+1:(c+1)*(self.tokenized_length)].contiguous()
			loss = self.cel(shift_logits, shift_labels)
			total_loss += loss
		mean_loss = total_loss / self.chunks
		return mean_loss, output

class ObjectiveMemoryTransformer(nn.Module, GenerationMixin):

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, objective='combined', compression=1, n_heads=4, n_chunks=4, fixed_memory=True, frozen_encoder=None, no_memory=False, decoder=None, blank_copy=False):
		super().__init__()

		self.no_memory = no_memory
		self.decoder_dim = dim
		if not self.no_memory:
			if frozen_encoder:
				for name, param in frozen_encoder.named_parameters():
					param.requires_grad = False
				self.encoder = frozen_encoder
			else:
				llama_config_kwargs = {
					'hidden_size': encoder_dim,
					'intermediate_size': 4*encoder_dim,
					'num_hidden_layers': depth,
					'num_attention_heads': n_heads,
					'vocab_size': n_vocab
				}
				encoder_configuration = LlamaConfig(**llama_config_kwargs)
				self.encoder = LlamaModel(encoder_configuration)
		else:
			self.encoder = None

		self.wte = nn.Embedding(n_vocab, encoder_dim) #NB: originally dim, does not matter for frozen encoder
		self.decoder_proj = None

		if decoder:
			if isinstance(decoder, PeftModelForCausalLM):
				self.decoder = decoder.base_model.model.model
				self.decoder_wte = decoder.base_model.model.model.embed_tokens
				self.lm_head = decoder.base_model.model.lm_head
			else:
				self.decoder = decoder.model
				self.decoder_wte = decoder.model.embed_tokens
				self.lm_head = decoder.lm_head
		else:
			llama_config_kwargs = {
				'hidden_size': dim,
				'intermediate_size': 4*dim,
				'num_hidden_layers': depth,
				'num_attention_heads': n_heads,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)
			self.decoder = LlamaModel(decoder_configuration)
			self.decoder_wte = nn.Embedding(n_vocab, dim)
			self.lm_head = nn.Linear(dim, n_vocab, bias=False)

		if encoder_dim != dim:
			self.decoder_proj = nn.Linear(encoder_dim, dim) # project if necessary

		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		self.chunks = n_chunks
		self.fixed_memory = fixed_memory
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			self.up = nn.Linear(encoder_dim//compression, encoder_dim)
		self.blank_copy = blank_copy
		assert objective in ['copy', 'combined', 'clm'], 'Unrecognized objective function'
		self.objective = objective

		self.generation_config = GenerationConfig()
		config  = {
				 'hidden_size': hidden_dim,
				 'intermediate_size': 4*hidden_dim,
				 'num_hidden_layers': num_blocks,
				 'num_attention_heads': 4, # mock heads
				 'vocab_size': vocab_size
			 }
		self.config = LlamaConfig(**config)
		self.main_input_name = 'input_ids'
		max_input_length = 2048
		generation_config_args = {'max_length': max_input_length}
		self.max_length = length
		self._supports_cache_class = False
		self.device = self.output_layer.weight.device
		self.generation_config = GenerationConfig(**generation_config_args)

	def can_generate(self):
		return True

	def _is_stateful(self):
		return False

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
		input_ids = input_ids.to(device)
		all_inputs = [[input_ids, labels]]
		
		total_loss = 0
		for input_ids, labels in all_inputs:
			# generate encoder embeddings
			embedding_array = []
			i = 0
			attention_chunks = []
			if self.no_memory:
				i = 1e9
				embedding_array = [torch.ones((input_ids.shape[0], 1, self.decoder_dim)).to(device) for _ in range(self.chunks)]

			while input_ids.shape[1] - self.tokenized_length > i:
				input_chunk, attention_chunk = self.wte(input_ids[:, i: i+self.tokenized_length]), attention_mask[:, i: i+self.tokenized_length]
				
				x = self.encoder(inputs_embeds=input_chunk, attention_mask=attention_chunk)
				if not torch.is_tensor(x):
					x = x.last_hidden_state
				
				encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
				if self.compression:
					encoder_embedding = self.down(encoder_embedding)
					encoder_embedding = self.up(encoder_embedding)
				if self.decoder_proj:
					encoder_embedding = self.decoder_proj(encoder_embedding)
				embedding_array.append(encoder_embedding)
				attention_chunks.append(attention_chunk)
				i += self.tokenized_length

			# embedding_array now stores length // n_ctx - 1 embeddings
			input_embeddings = self.decoder_wte(input_ids)
			all_outputs = []
			for c in range(self.chunks): 
				decoder_embeds = input_embeddings[:, (c*self.tokenized_length):(c+1)*self.tokenized_length]
				if self.fixed_memory:
					pad = torch.ones((input_ids.shape[0], self.chunks-c, input_embeddings.shape[2])).to(device)
					x = torch.cat((embedding_array[:c] + [pad] + [decoder_embeds]), dim=1) # concatenation on token dim
				else:
					x = torch.cat((embedding_array[:c] + [decoder_embeds]), dim=1) # concatenation on token dim
				if attention_mask is not None:
					attention_mask = torch.cat((torch.ones(input_ids.shape[0], c).to(device), attention_mask), dim=1)
				# feed pre-concatenated input embeddings to the transformer decoder
				x = self.decoder(inputs_embeds=x, attention_mask=attention_mask)
				output = self.lm_head(x.last_hidden_state)

				if labels is not None and labels.dim() > 2:
					labels = rearrange(labels, 'b p t -> b (p t)')
				output = rearrange(output, 'b t e -> b e t')
				
				if self.fixed_memory:
					all_outputs.append(output[..., self.chunks:self.chunks+self.tokenized_length]) # assemble all outputs
					shift_logits = output[..., self.chunks:self.chunks+self.tokenized_length-1].contiguous()
				else:
					all_outputs.append(output[..., c:]) # assemble all outputs
					shift_logits = output[..., c:-1].contiguous() # first c 'tokens' are encoding

				if labels is not None:
					shift_labels = labels[..., (c*self.tokenized_length)+1:(c+1)*(self.tokenized_length)].contiguous()
				# only take loss of non-fully-masked blocks
				if torch.all(shift_labels == -100):
		 			continue
				loss = self.cel(shift_logits, shift_labels)
				if not torch.isnan(loss):
					total_loss += loss
			if total_loss == 0:
				total_loss = loss # if no chunks are valid

		mean_loss = total_loss / self.chunks
		all_outputs = torch.cat(all_outputs, dim=2) # concat in token dim
		truncated_logits = all_outputs[:, :input_length, :] # truncate logits (for padded inputs)
		if labels is not None:
			loss = 0
		else:
			loss = mean_loss
		CausalLMOutput(loss=loss, logits=truncated_logits)