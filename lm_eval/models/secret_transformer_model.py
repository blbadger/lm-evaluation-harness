import os
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, LlamaForCausalLM, BertConfig, BertForMaskedLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin, GenerationConfig

import torch._dynamo
torch._dynamo.config.suppress_errors = True

class ParallelNoninvertibleModel(nn.Module, GenerationMixin):
       
    def __init__(self, 
        n_vocab, 
        dim, 
        provider_model, 
        inversion_decoder, 
        inversion_head=None, 
        clm_head=None, 
        tokenized_length=512, 
        freeze_decoders=True, 
        clm_loss_only=False,
        parallel_encoder=None,
        unified_decoder=None,
        unified_encoder=None,
        n_tokens_obfuscated=None,
        no_provider_modules=False,
        compress_provider_factor=1,
        route_method='embedding_split',
        mask_secret_tokens=False,
        embeddings_to_unroll=1
    ):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()
        self.tokenized_length = tokenized_length
        if not n_tokens_obfuscated:
            self.obfuscate_first_n = self.tokenized_length
        else:
            self.obfuscate_first_n = n_tokens_obfuscated
        self.dim = dim
        self.inversion_head = nn.Linear(dim, n_vocab)
        self.clm_head = nn.Linear(dim, n_vocab)

        self.inversion_decoder = inversion_decoder
        
        self.n_vocab = n_vocab
        self.clm_head = nn.Linear(dim, n_vocab)
        # for split embeddings
        self.client_proj = nn.Linear(dim//2, dim)
        self.provider_proj = nn.Linear(dim//2, dim)
        self.dim = dim

        self.clm_loss_only = clm_loss_only
        # for parallel modeling
        self.unified_encoder = unified_encoder # LlamaModel
        self.parallel_encoder = parallel_encoder # LlamaModel 
        self.unified_decoder = unified_decoder # LlamaModel
        self.provider_model = provider_model
        self.no_provider_modules = no_provider_modules

        for _, param in self.inversion_decoder.named_parameters():
            param.requires_grad = False 
        for _, param in self.parallel_encoder.named_parameters():
            param.requires_grad = True
        for _, param in self.unified_decoder.named_parameters():
            param.requires_grad = True
        for _, param in self.unified_encoder.named_parameters():
            param.requires_grad = True

        self.encoder_dim = self.unified_encoder.config.hidden_size
        self.unified_decoder_dim = self.unified_decoder.config.hidden_size
        self.clm_head = nn.Linear(self.unified_decoder_dim, n_vocab) # TODO: replace overwrite here
        self.route_method = route_method
        if self.route_method == 'unroll_embedding':
            self.unroll_projection = nn.Linear(self.encoder_dim//2, self.encoder_dim)
        self.provider_emb_compression = compress_provider_factor
        if self.provider_emb_compression > 1:
            self.in_provider_proj = nn.Linear(dim, dim//self.provider_emb_compression)
            self.out_provider_proj = nn.Linear(dim//self.provider_emb_compression, dim)
        self.mask_secret_tokens = mask_secret_tokens

        self.provider_in_proj = None
        self.provider_out_proj = None
        encoder_dim = self.unified_encoder.config.hidden_size
        if self.unified_encoder.config.hidden_size != self.provider_model.config.hidden_size:
            self.provider_in_proj = nn.Linear(self.unified_encoder.config.hidden_size, self.provider_model.config.hidden_size)

        if self.provider_model.config.hidden_size != self.unified_decoder.config.hidden_size:
            self.provider_out_proj = nn.Linear(self.provider_model.config.hidden_size, self.unified_decoder.config.hidden_size)

        
        self.embeddings_to_unroll = embeddings_to_unroll
        self.provider_in_proj = None
        self.provider_out_proj = None
        if self.unified_encoder.config.hidden_size != self.provider_model.config.hidden_size:
            self.provider_in_proj = nn.Linear(self.unified_encoder.config.hidden_size, self.provider_model.config.hidden_size)

        if self.provider_model.config.hidden_size != self.unified_decoder.config.hidden_size:
            self.provider_out_proj = nn.Linear(self.provider_model.config.hidden_size, self.unified_decoder.config.hidden_size)

        self.generation_config = GenerationConfig()
        config  = {
                 'hidden_size':dim,
                 'intermediate_size': 4*dim,
                 'num_hidden_layers': 16,
                 'num_attention_heads': 4,
                 'vocab_size': n_vocab
                }
        self.config = LlamaConfig(**config)
        self.main_input_name = 'input_ids'
        self._supports_cache_class = False
        self.cache_built = False
        self.device = self.clm_head.weight.device

    def can_generate(self):
        return True

    def _is_stateful(self):
        # no caching implemented, not necessary for limited gens
        return False

    def unroll_embedding(self, embedding):
        embedding_stack = []
        # sliding window unroll over hidden dim
        for i in range(self.obfuscate_first_n):
            i %= self.encoder_dim
            sliding_window = embedding[..., i:i+self.encoder_dim//2]
            if i+self.encoder_dim//2 > self.encoder_dim:
                residual = i+self.encoder_dim//2 - self.encoder_dim 
                # loop around to first index
                sliding_window = torch.cat((sliding_window, embedding[..., :residual]), dim=2)
            embedding_stack.append(sliding_window)
        embedding = torch.cat(embedding_stack, dim=1)
        embedding = self.unroll_projection(embedding)
        return embedding

    def mask_obfuscated_tokens(self, embeddings):
        mask_embedding = torch.zeros((embeddings.shape[0], self.obfuscate_first_n, embeddings.shape[1])).to(embeddings.dtype).to(embeddings.device)
        embeddings[:, :self.obfuscate_first_n, :] = mask_embedding # [b t e]
        return embeddings

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        x = input_ids
        
        if x.shape[1] < 256:
            pad_size = 256 - x.shape[1]
            pad_tokens = torch.ones((x.shape[0], pad_size)).to(x.device).to(x.dtype)
            x = torch.cat((pad_tokens, x), dim=1)
            
        # Attention mask is not properly passed, we need to pass ourselves
        attention_mask = torch.ones(x.shape).to(input_ids.device)
        attention_mask = torch.where(x != 1, attention_mask, 0)
        attention_mask = torch.where(x != 0, attention_mask, 0)
       
        encoder_outputs = self.unified_encoder(input_ids=x, attention_mask=attention_mask).last_hidden_state # shape [b t e]
        
        if self.route_method == 'embedding_split':
            client_input = self.client_proj(encoder_outputs[:, :, :self.encoder_dim//2])
            provider_input = self.provider_proj(encoder_outputs[:, :, self.encoder_dim//2:])

        elif self.route_method == 'unroll_embedding':
            client_input = encoder_outputs
            # unroll last obfuscated embedding and concat with all others
            unrolled_embedding = torch.zeros((client_input.shape[0], self.obfuscate_first_n, client_input.shape[2])).to(client_input.device).to(client_input.dtype)
            for m in range(self.embeddings_to_unroll):
                unrolled_embedding += self.unroll_embedding(encoder_outputs[:, self.obfuscate_first_n-(m+1), :].unsqueeze(1))
            provider_input = torch.cat((unrolled_embedding, encoder_outputs[:, self.obfuscate_first_n:, :]), dim=1)

        if self.provider_in_proj is not None:
            provider_input = self.provider_in_proj(provider_input)

        if self.provider_emb_compression > 1:
            prefix_provider_input = self.out_provider_proj(self.in_provider_proj(provider_input[:, :self.obfuscate_first_n, :]))
            provider_input = torch.cat((prefix_provider_input, provider_input[:, self.obfuscate_first_n:, :]), dim=1) # cat along token dim

        if self.mask_secret_tokens:
            provider_input = self.mask_obfuscated_tokens(provider_input)

        provider_output = self.provider_model(inputs_embeds=provider_input).last_hidden_state

        if self.provider_out_proj is not None:
            provider_output = self.provider_out_proj(provider_output)

        parallel_x = self.parallel_encoder(inputs_embeds=client_input).last_hidden_state
        if self.provider_out_proj is not None:
            provider_output = self.provider_out_proj(provider_output)

        if self.no_provider_modules:
            combined_output = parallel_x # omits the provider modules, negative control
        else:
            combined_output = parallel_x + provider_output

        clm_x = self.unified_decoder(inputs_embeds=combined_output).last_hidden_state
        logits = self.clm_head(clm_x) # shape b t e 

        if labels is not None:
            output = rearrange(logits, 'b t e -> b e t').clone()
            shift_logits = output[..., self.obfuscate_first_n:-1]
            shift_labels = labels.to(device)[..., self.obfuscate_first_n+1:]
            clm_loss = self.cel(shift_logits, shift_labels) 
        else:
            clm_loss = 0
        
        return CausalLMOutput(loss=clm_loss, logits=logits) # requires b t e shape
        


class SecretDecoder(nn.Module):

    def __init__(self, n_vocab, dim, model, tokenized_length=512, embedding_dim=512, reduce_loss=True):
        super().__init__()
        self.model = model # assumes a LlamaModel
        
        self.tokenized_length = tokenized_length
        self.in_proj = None
        if not reduce_loss:
            self.cel = nn.CrossEntropyLoss(reduction='none')
        else:
            self.cel = nn.CrossEntropyLoss()

        if embedding_dim != dim:
            self.in_proj = nn.Linear(embedding_dim, dim)

    def forward(self, inputs_embeds, labels=None):
        x = inputs_embeds
        # x is [b t e]
        if x.dim() > 3:
            x = x.to(device).squeeze(1)
        if self.in_proj:
            x = self.in_proj(x)
        x = self.model(inputs_embeds=x).logits

        # no token shift
        output = rearrange(x, 'b t e -> b e t')
        if labels is not None:
            loss = self.cel(output, labels)
            return loss, output
        return output

def init_noninvertible_parallelmodel(
    tokenizer, 
    vocab_size, 
    n_tokens_obfuscated,
    compress_provider_factor=1,
    route_method='embedding_split',
    context_length=1024, 
    decoder_dim=512, 
    client_dim=512,
    inverter_layers=8, 
    unified_encoder_layers=2,
    provider_model_layers=16, 
    client_encoder_layers=2,
    unified_decoder_layers=2,
    n_heads=4,
    embeddings_to_unroll=1
    ):
    # inversion model specification: must be bidirectional if the input token indices are not reversed
    config_kwargs = {
        'hidden_size': decoder_dim,
        'intermediate_size': 4 * decoder_dim,
        'num_hidden_layers': inverter_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length,
    }
    configuration = BertConfig(**config_kwargs)
    model = BertForMaskedLM(configuration)
    inverter = SecretDecoder(vocab_size, decoder_dim, model, embedding_dim=decoder_dim, reduce_loss=False)

    # unified encoder specification
    config_kwargs = { 
        'hidden_size': client_dim,
        'intermediate_size': 4*client_dim,
        'num_hidden_layers': unified_encoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    encoder_configuration = LlamaConfig(**config_kwargs)
    unified_encoder = LlamaModel(encoder_configuration)

    # client encoder specification
    config_kwargs = { 
        'hidden_size': client_dim,
        'intermediate_size': 4*client_dim,
        'num_hidden_layers': client_encoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    client_encoder = LlamaModel(configuration)

    # provider model specification
    config_kwargs = { 
        'hidden_size': decoder_dim,
        'intermediate_size': 4*decoder_dim,
        'num_hidden_layers': provider_model_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    provider_model = LlamaModel(configuration)

    # unified decoder model specification
    config_kwargs = { 
        'hidden_size': client_dim,
        'intermediate_size': 4*client_dim,
        'num_hidden_layers': unified_decoder_layers,
        'num_attention_heads': n_heads,
        'vocab_size': vocab_size,
        'max_position_embeddings': context_length
    }

    configuration = LlamaConfig(**config_kwargs)
    unified_decoder = LlamaModel(configuration)

    model = ParallelNoninvertibleModel(
        vocab_size, 
        decoder_dim, 
        provider_model, 
        inverter, 
        tokenized_length=context_length, 
        clm_loss_only=False,
        parallel_encoder=client_encoder,
        unified_decoder=unified_decoder,
        unified_encoder=unified_encoder,
        n_tokens_obfuscated=n_tokens_obfuscated,
        compress_provider_factor=compress_provider_factor,
        route_method=route_method,
        embeddings_to_unroll=embeddings_to_unroll
    )
    return model, inverter

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')

    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(f'{data_root}/tokenizer_fineweb_8k')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    n_tokens_obfuscated = 256
    compress_provider_factor = 1
    route_method = 'unroll_embedding'
    client_dim = 1024
    decoder_dim = 1024
    embeddings_to_unroll = 1
    model, inverter = init_noninvertible_parallelmodel(tokenizer, 
        vocab_size, 
        n_tokens_obfuscated, 
        decoder_dim=decoder_dim, 
        compress_provider_factor=compress_provider_factor, 
        route_method=route_method, 
        client_dim=client_dim, 
        embeddings_to_unroll=embeddings_to_unroll
    )
