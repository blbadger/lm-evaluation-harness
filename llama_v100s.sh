accelerate launch -m lm_eval --model hf \
    --model_args pretrained=/home/bbadger/Desktop/gsm8k_transformer_s5_b15x4/checkpoint-1866 \
    --tasks gsm8k_pass@50 \
    --batch_size 64
