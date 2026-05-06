accelerate launch -m lm_eval --model tree-srm \
    --model_args pretrained=/home/bbadger/Desktop/gsq_4_mixed_decay_nonparallel_projs_k1_1024_n16_c1024_b16x4/checkpoint-600000/model.safetensors \
    --model_args reward_pretrained=/home/bbadger/Desktop/gsm8k_reward_model_t512_gsqinit/checkpoint-80000/model.safetensors \
    --model_args tree_size=512 \
    --tasks gsm8k_tree_pass \
    --batch_size 4096

