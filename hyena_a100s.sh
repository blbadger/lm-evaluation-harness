lm_eval --model hyena \
    --model_args pretrained=/home/badger/fineweb_hyena_128_n16_b32x4/checkpoint-200000/pytorch_model.bin \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
