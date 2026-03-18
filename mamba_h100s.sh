accelerate launch -m lm_eval --model mamba-clm \
    --model_args pretrained=/home/azureuser/finemath_mamba_cache_secondrun_256_s128_n16_c512_b64x2/checkpoint-200000/model.safetensors \
    --tasks arc_easy,glue
    --batch_size 128
