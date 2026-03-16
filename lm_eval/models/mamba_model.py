import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import MambaConfig, Mamba2Config, MambaForCausalLM, Mamba2ForCausalLM, Mamba2Model
from transformers import AutoTokenizer, LlamaConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin, GenerationConfig
import datasets
from datasets import load_from_disk
import mlflow
from prettytable import PrettyTable
import os
from dotenv import load_dotenv
import shutil

class MambaCLM(nn.Module):
   
   def __init__(self, model, dim, vocab_size):
       super().__init__()
       self.model = model
       self.lm_head = nn.Linear(dim, vocab_size)
       self.vocab_size = vocab_size
       self.loss_fn = nn.CrossEntropyLoss()

       config_kwargs = {
            'hidden_size': dim,
            'intermediate_size': 4*dim,
            'num_hidden_layers': n_layers,
            'num_attention_heads': num_heads,
            'vocab_size': vocab_size,
            'state_size': state_size,
            'hidden_dropout_prob': 0,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'chunk_size': context_length,
            'num_heads': num_heads,
            'head_dim': head_dim
        }

        self.config = Mamba2Config(**config_kwargs)
        self.main_input_name = 'input_ids'
        max_input_length = 1024
        generation_config_args = {'max_length': max_input_length}
        self.generation_config = GenerationConfig(**generation_config_args)
        self.max_length = 1024
        self._supports_cache_class = False
        self.device = self.output_layer.weight.device

    def can_generate(self):
        return True

    def _is_stateful(self):
        return False
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

   def forward(self, input_ids, labels=None, **kwargs):
        labels = labels[:, 1:].contiguous()
        x = self.model(input_ids, use_cache=True).last_hidden_state
        logits = self.lm_head(x)
        logits_out = logits[:, :-1].contiguous()
        
        if labels is not None:
            logits = logits[:, :-1].contiguous()
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)

            loss = self.loss_fn(logits, labels)
            return CausalLMOutput(loss=loss, logits=logits_out)

        else:
            return CausalLMOutput(loss=0, logits=logits_out)