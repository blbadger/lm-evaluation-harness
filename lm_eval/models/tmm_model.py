import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin, GenerationConfig
import datasets
from datasets import load_from_disk
from safetensors.torch import load_model
import os
from dotenv import load_dotenv
import shutil

class ToeplitzCausalLinear(nn.Module):
    """
    A linear layer with a triangular (causal) mask applied to the weight matrix.
    This ensures each position i cannot use info from positions > i.
    """

    def __init__(self, dim: int):

        super().__init__()

        # Standard weight + bias
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        Given a vector v of shape (m,), returns an (m x m) matrix M
        where M[i, j] = v[j - i] if j >= i, and 0 otherwise.

        For example, if v = [a, b, c, d] then M will be:

        [ a  b  c  d ]
        [ 0  a  b  c ]
        [ 0  0  a  b ]
        [ 0  0  0  a ]
        """
        v = v.reshape(-1)  # Ensure v is a 1D tensor
        m = v.shape[0]
        # Create index grids for rows and columns
        i, j = torch.meshgrid(
            torch.arange(m, device=v.device),
            torch.arange(m, device=v.device),
            indexing="ij",
        )
        # j - i gives the offset into v. When j < i, we want a 0.
        M = torch.where(
            j >= i, v[j - i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
        )
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, embed_dim, seq_len)
        """
        B, E, S = x.shape
        W = self.vector_to_matrix(self.weight)
        x_reshaped = x.reshape(B * E, S)  # (B*E, S)
        out = x_reshaped @ W  # (B*E, S)
        out = out + self.bias  # broadcast bias
        out = out.view(B, E, S)  # reshape back
        return out


class ToeplitzHeads(nn.Module):

    def __init__(
        self,
        dim: int,
        seq_len: int,
        hidden_dim: int,
        n_heads: int,
        expanded_convs: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.proj_head = nn.ModuleList(
            [nn.Linear(dim, hidden_dim) for i in range(n_heads)]
        ).to(device)

        self.out_proj = nn.Linear(dim, dim)

        if expanded_convs:
            self.mixer_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        ToeplitzCausalLinear(seq_len),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                        ToeplitzCausalLinear(seq_len),
                    )
                    for i in range(n_heads)
                ]
            )
        else:
            self.mixer_heads = nn.ModuleList(
                [ToeplitzCausalLinear(seq_len) for i in range(n_heads)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        x = rearrange(x, "b e t -> b t e")
        # pre-concatenated out projection
        for head in range(self.n_heads):
            projection = self.proj_head[head](x)
            projection = rearrange(projection, "b t e -> b e t")
            conv_projection = self.mixer_heads[head](projection)
            rearranged_conv = rearrange(conv_projection, "b e t -> b t e")
            activations.append(rearranged_conv)

        # concatenate and project multi-headed output
        hidden_layer = torch.cat(activations, dim=2)
        hidden_layer = self.out_proj(hidden_layer)
        hidden_layer = rearrange(hidden_layer, "b t e -> b e t")
        return hidden_layer


class MixerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        expansion_factor: int = 4,
        dropout: float = 0.,
        heads=None,
        expanded_convs=False,
    ):

        super(MixerBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.expansion_factor = expansion_factor

        # channel-norm
        self.channel_norm = nn.LayerNorm(hidden_dim)

        # channel-mixing layer
        self.channel_mixing_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

        # token-norm
        self.token_norm = nn.LayerNorm(hidden_dim)
        if heads and heads > 0:
            self.token_mixing_layer = ToeplitzHeads(
                hidden_dim,
                seq_len,
                hidden_dim // heads,
                heads,
                expanded_convs=expanded_convs,
            )  # type: ignore[assignment]

        else:

            if expanded_convs:
                # token-mixing layer
                self.token_mixing_layer = nn.Sequential(
                    ToeplitzCausalLinear(seq_len),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    ToeplitzCausalLinear(seq_len),
                )  # type: ignore[assignment]

            else:
                # flat mixer layer
                self.token_mixing_layer = ToeplitzCausalLinear(seq_len)  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.channel_norm(x)
        x = self.channel_mixing_layer(x)
        x = x + res

        res = x
        x = self.token_norm(x)
        x = x.transpose(1, 2)
        x = self.token_mixing_layer(x)
        x = x.transpose(1, 2)
        x = x + res
        return x


class MLPMixer(nn.Module, GenerationMixin):

	def __init__(
		self,
		vocab_size: int,
		hidden_dim: int,
		seq_len: int,
		num_blocks: int,
		heads=None,
		kernel=1,
		expanded_convs=False,
	):

		super(MLPMixer, self).__init__()

		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		self.seq_len = seq_len
		self.num_blocks = num_blocks
		self.input_layer = nn.Embedding(vocab_size, hidden_dim)

		self.mixer_blocks = nn.ModuleList(
			[
				MixerBlock(
					hidden_dim, seq_len, heads=heads, expanded_convs=expanded_convs
				)
				for _ in range(num_blocks)
			]
		)
		self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False, device='cuda')

		self._init_weights()
		self.loss_fn = nn.CrossEntropyLoss()

		self.generation_config = GenerationConfig()
		config  = {
				 'hidden_size':hidden_dim,
				 'intermediate_size': 4*hidden_dim,
				 'num_hidden_layers': layers,
				 'num_attention_heads': n_heads,
				 'vocab_size': vocab_size
			 }
		self.config = LlamaConfig(**config)
		self.main_input_name = 'input_ids'
		max_input_length = 2048
		generation_config_args = {'max_length': max_input_length}
		self.generation_config = GenerationConfig(**generation_config_args)
		self.max_length = 1024
		self._supports_cache_class = False
		self.device = self.output_layer.weight.device

	def can_generate(self):
		return True

	def _is_stateful(self):
		return False
	

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, ToeplitzCausalLinear):
				# Kaiming He initialization for Swish activation
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def count_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def forward(self, input_ids, labels=None, **kwargs):
		# pad input_ids
		pad_sizes = [self.seq_len - len(input_ids[i]) for i in range(len(input_ids))]
		input_lengths = [len(input_ids[i]) for i in range(len(input_ids))]

		padded_inputs = []
		for i in range(len(input_ids)):
			input_id_arr = input_ids[i].unsqueeze(0)
			pad = torch.ones(pad_sizes[i], dtype=torch.long).to(input_ids.device).unsqueeze(0)
			padded_input = torch.cat((input_id_arr, pad), dim=1)
			padded_inputs.append(padded_input)
		input_ids = torch.cat(padded_inputs, dim=0)
		labels = torch.where(input_ids==1, -100, input_ids) #mask pad token loss

		if labels is not None:
			labels = labels[:, 1:].contiguous()

		# model's forward pass
		x = self.input_layer(input_ids)
		for block in self.mixer_blocks:
			x = block(x)
		logits = self.output_layer(x)
		logits = logits[:, :-1].contiguous()

		truncated_logits = []
		for i in range(logits.shape[0]):
			truncated_logits.append(logits[i, :input_lengths[i]])
		truncated_logits = torch.stack(truncated_logits, dim=0)

		if labels is not None:
			logits = logits.view(-1, self.vocab_size)
			labels = labels.view(-1)

			loss = self.loss_fn(logits, labels)
			return CausalLMOutput(loss=loss, logits=truncated_logits)

		else:
			return CausalLMOutput(loss=0, logits=truncated_logits)