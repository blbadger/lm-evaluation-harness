accelerate launch -m lm_eval --model mamba-default \
    --model_args pretrained=/home/azureuser/fineweb_mamba_cache_256_s128_n16_c1024_b32x2/checkpoint-64000/model.safetensors \
    --tasks arc_easy\
    --batch_size 128
