accelerat launch -m lm_eval --model hf \
    --model_args pretrained=/home/azureuser/fineweb_transformer_d512_n16_c1024_b64x2/checkpoint-200000 \
    --tasks gsm8k_pass \
    --batch_size 128
