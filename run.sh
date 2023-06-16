set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_en.py 'rope_imp__2d_raw_post' 4 'rope with post-attn-norm +imp(no bias div)'
