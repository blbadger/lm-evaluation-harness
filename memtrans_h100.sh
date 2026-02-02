accelerate launch -m lm_eval --model memtrans \
    --model_args pretrained=/home/azureuser/ \
    --tasks squad_completion \
    --device cuda:0 \
    --batch_size 128
