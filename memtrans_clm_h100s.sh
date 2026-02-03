lm_eval --model memtrans \
    --model_args pretrained=/home/azureuser/fineweb_clm_fullcurriculumpretrained_memtrans_c256x4_512c1_d512_n16_c256_b32x2x1/checkpoint-100000/model.safetensors \
    --device cuda:0 \
    --tasks arc_easy,glue,hellaswag,lambada_openai \
    --batch_size 128
