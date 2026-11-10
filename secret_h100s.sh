lm_eval --model secrettrans \
    --model_args pretrained=/home/azureuser/noninvertible_parallelmodel_unroll_s256_2p16c2u2_d512_b32x2/step_200000/clm_model.safetensors \
    --device cuda:1 \
    --tasks ifeval,squadv2,xwinograd,gsm8k,arc_easy,wikitext,glue,lambada_openai,hellaswag \
    --batch_size 256
