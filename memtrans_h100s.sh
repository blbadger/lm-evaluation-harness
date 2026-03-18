lm_eval --model memtrans \
    --model_args pretrained=/home/azureuser/fineweb_combined_curriculum_blankcopy_frozenauto_memtrans_512c1_d512_n16_c256_b32x2x1/checkpoint-100000/model.safetensors \
    --device cuda:1 \
    --tasks longbench,swde \
    --batch_size 128
