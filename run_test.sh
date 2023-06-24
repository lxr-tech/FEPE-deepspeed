set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_rope_inv_2d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_rope_inv_2d_log'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_rope_inv_1d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_rope_inv_1d_log'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_rope_imp_2d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_rope_imp_2d_log'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_rope_imp_1d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_rope_imp_1d_log'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_xpos_inv_2d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_xpos_inv_2d_log'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_xpos_inv_1d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_xpos_inv_1d_log'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_xpos_imp_2d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_xpos_imp_2d_log'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_xpos_imp_1d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_test_en.py 'pre_xpos_imp_1d_log'
