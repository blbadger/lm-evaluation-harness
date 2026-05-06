accelerate launch -m lm_eval --model srm \
    --model_args pretrained=/home/bbadger/Desktop/gsq_4_mixed_decay_nonparallel_projs_k1_1024_n16_c1024_b16x4/checkpoint-592000/model.safetensors \
    --tasks gsm8k_pass@50 \
    --batch_size 16384
