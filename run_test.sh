set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'xpos_inv_2d_raw'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'xpos_inv_2d_log'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'xpos_inv_1d_raw'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'xpos_inv_1d_log'

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'xpos_imp_2d_raw'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'xpos_imp_2d_log'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'xpos_imp_1d_raw'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'xpos_imp_1d_log'

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'rope_inv_2d_raw'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'rope_inv_2d_log'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'rope_inv_1d_raw'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'rope_inv_1d_log'

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'rope_imp_2d_raw'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'rope_imp_2d_log'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'rope_imp_1d_raw'
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_test_en.py 'rope_imp_1d_log'

