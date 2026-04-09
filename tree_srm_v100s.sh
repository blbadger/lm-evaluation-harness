accelerate launch -m lm_eval --model tree-srm \
    --model_args pretrained=/home/bbadger/Desktop/gsm8k_SFT_srm_c1024/chkpt-300/model.safetensors \
    --model_args reward_pretrained=/home/bbadger/Desktop/gsm8k_tree_reward_b512_continued/checkpoint-2600/model.safetensors \
    --model_args tree_size=512 \
    --tasks gsm8k_tree_pass_50 \
    --batch_size 2000

