from hyena_model import HyenaModel
from transformers import AutoTokenizer
from lm_eval.api.model import LM
import os
from dotenv import load_dotenv
import torch

@register_model("custom_hyena")
class MyCustomLM(LM):

    def __init__(self, dim=512, n_ctx=1024):
        super().__init__()
        load_dotenv()
        checkpoint_root = os.getenv('CHECKPOINT_ROOT')
        data_root = os.getenv('DATA_ROOT')
        model_path = os.join(data_root, 'fineweb_hyena_512_n16_c1024_b32x2/checkpoint-200000/pytorch_model.bin')
        self.tokenizer = AutoTokenizer.from_pretrained(data_root + '/tokenizer_fineweb_8k')
        n_vocab = len(tokenizer) # 8000
        dim = dim
        depth = 16
        length = n_ctx
        model = HyenaModel(n_vocab, dim, depth, length)
        model.load_state_dict(torch.load(model_path))
        self.model

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        

    def generate_until(self, requests: list[Instance]) -> list[str]:
 