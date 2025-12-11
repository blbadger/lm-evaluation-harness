lm_eval --model hyena \
    --model_args pretrained=/home/azureuser/fineweb_toep_1024_c1024.safetensors \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 128
