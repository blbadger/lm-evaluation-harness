lm_eval --model tmm \
    --model_args pretrained=/home/azureuser/fineweb_toep_1024_c512.safetensors \
    --tasks squad_completion \
    --device cuda:0 \
    --batch_size 128
