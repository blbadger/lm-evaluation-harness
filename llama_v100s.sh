accelerate launch -m lm_eval --model hf \
    --model_args pretrained=/home/bbadger/Desktop/fineweb_training/fineweb_llama_512_c1024/checkpoint-196000 \
    --tasks gsm8k_pass \
    --batch_size 128
