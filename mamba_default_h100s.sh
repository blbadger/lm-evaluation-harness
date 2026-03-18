accelerate launch -m lm_eval --model mamba_ssm \
    --model_args pretrained=/home/azureuser/fineweb_mamba_cache_256_s128_n16_c1024_b32x2/checkpoint-64000 \
    --tasks wikitext\
    --batch_size 128
