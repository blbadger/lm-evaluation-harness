from hyena_model import HyenaModel
from transformers import AutoTokenizer
from lm_eval.api.model import LM
import torch.nn.functional as F
import os
from dotenv import load_dotenv
import torch
from lm_eval.api.registry import register_model

@register_model("hyena_sparse")
class MyCustomLM(LM):

    def __init__(self, dim=512, n_ctx=1024):
        super().__init__()
        load_dotenv()
        checkpoint_root = os.getenv('CHECKPOINT_ROOT')
        data_root = os.getenv('DATA_ROOT')
        model_path = os.join(data_root, 'fineweb_hyena_512_n16_c1024_b32x2/checkpoint-200000/pytorch_model.bin')
        tokenizer = AutoTokenizer.from_pretrained(data_root + '/tokenizer_fineweb_8k')
        self.tokenizer = tokenizer
        n_vocab = self.tokenizer.vocab_size # 8000
        dim = dim
        depth = 16
        length = n_ctx
        model = HyenaModel(n_vocab, dim, depth, length)
        model.load_state_dict(torch.load(model_path))
        self.stopping_criteria = set(tokenizer.eos_token_id, tokenizer.bos_token_is, tokenizer.pad_token_id)
        self.model = model
        self.max_length = length

    @torch.no_grad()
    def _model_call(self, input_ids):
        return self.model(input_ids)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        contexts, continuations = requests.args
        context_ids = self.tokenizer.encode(contexts, return_tensors='pt', truncation='max_length', max_length = self.n_ctx)
        continuation_ids = self.tokenizer.encode(contexts, return_tensors='pt', truncation='max_length', max_length = self.n_ctx)
        #TODO: catenate contexts and continuations, form loglikelihoods over continuations
        logits = self._model_call(input_ids)[..., -1]
        log_likelihoods = F.log_softmax(logits, dim=-1)
        return (log_likelihoods, [False for i in range(len(log_likelihoods))])

        
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        # roll contexts
        contexts, gen_kwargs = requests.args

        

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        context, gen_kwargs = requests.args
        input_ids = self.tokenizer.encode(requests['string'])
        self.model.generate(input_ids, **gen_kwargs)
        output_string = self.tokenizer.decode(input_ids)
        return output_string
