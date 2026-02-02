lm_eval --model tmm \
    --model_args pretrained=/home/azureuser/fineweb_toep_1024_c1024.safetensors \
    --tasks longbench \
    --device cuda:0 \
    --batch_size 128
