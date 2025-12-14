accelerate launch -m lm_eval --model tmm \
    --model_args pretrained=/home/bbadger/Desktop/fineweb_toep_1024_n16_c1024_lpad_b16x4/checkpoint-200000 \
    --tasks squad_completion \
    --batch_size 128
