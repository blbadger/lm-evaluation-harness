lm_eval --model hyena \
    --model_args pretrained=/home/azureuser/fineweb_hyena_512_n16_c1024_b32x2/checkpoint-200000/pytorch_model.bin \
    --tasks longbench \
    --device cuda:0 \
    --batch_size 32
