lm_eval --model tmm \
    --model_args pretrained=/home/bbadger/Desktop/fineweb_toep_1024_n16_c1024_lpad_b16x4/checkpoint-200000/model.safetensors \
    --trust_remote_code True \
    --tasks lambada_openai \
    --device cuda:0 \
    --batch_size 32
