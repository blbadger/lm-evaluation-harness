accelerate launch -m lm_eval --model srm \
    --model_args pretrained=/home/bbadger/Desktop/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors \
    --tasks gsm8k \
    --batch_size 128
