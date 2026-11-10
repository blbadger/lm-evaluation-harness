accelerate launch -m lm_eval --model mamba-clm \
    --model_args pretrained=/home/azureuser/finemath_mamba_cache_secondrun_512_s256_n16_c512_b64x2/checkpoint-200000/model.safetensors \
    --tasks glue,swde,wikitext,squad_completion,lambada_openai\
    --batch_size 256
