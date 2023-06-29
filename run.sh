set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_en.py 'init_post_both_xpos_imp_2d_raw' 4 'xpos +imp' > clm_init_post_both_xpos_imp_2d_raw_1_512.log 2>&1
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_en.py 'init_post_both_xpos_imp_1d_raw' 4 'xpos +imp+1d' > clm_init_post_both_xpos_imp_1d_raw_1_512.log 2>&1
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_en.py 'init_post_both_xpos_imp_2d_log' 4 'xpos +imp+log' > clm_init_post_both_xpos_imp_2d_log_1_512.log 2>&1
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_en.py 'init_post_both_xpos_imp_1d_log' 4 'xpos +imp+1d+log' > clm_init_post_both_xpos_imp_1d_log_1_512.log 2>&1
