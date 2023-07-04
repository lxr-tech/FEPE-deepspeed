set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 llama_test_pe.py 'init_pre_rope_inv_2d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 llama_test_pe.py 'init_pre_rope_inv_2d_log'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 llama_test_pe.py 'init_pre_xpos_inv_2d_raw'
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 llama_test_pe.py 'init_pre_xpos_inv_2d_log'
