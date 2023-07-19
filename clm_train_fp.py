import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

from utils.base import setup_seed
from utils.clm_tools import MyDataset, MyDataloader, get_dataset_info, DataCollatorForCausalLM
from utils.clm_trainer import TrainerForCausalLM
from configs.clm_train_config import model_args, train_args

# import deepspeed
# from transformers.deepspeed import HfDeepSpeedConfig

from models.llama_with_pe_fp import LlamaForCausalLM

import os
import argparse

parser = argparse.ArgumentParser(description='define fp config')
parser.add_argument('--dim', type=bool, default=False)
parser.add_argument('--exp', type=bool, default=False)
parser.add_argument('--imp', type=bool, default=False)
parser.add_argument('--log', type=bool, default=False)

parser.add_argument('--flash_train', type=bool, default=False)
parser.add_argument('--flash_test', type=bool, default=True)
parser.add_argument('--post_norm_attn', type=bool, default=False)
parser.add_argument('--post_norm_ffn', type=bool, default=False)
parser.add_argument('--init', type=bool, default=True)

parser.add_argument('--default', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
parser.add_argument('--pe_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
parser.add_argument('--qk_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
parser.add_argument('--vo_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
parser.add_argument('--ffn_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
parser.add_argument('--norm_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])

parser.add_argument('--key', type=str, default='')
parser.add_argument('--modification', type=str, default='')

# # https://deepspeed.readthedocs.io/en/latest/initialize.html
#
# parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
# parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()

fp = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}

fp_config = {'default': fp[args.default], 'qk_fp': fp[args.qk_fp], 'vo_fp': fp[args.vo_fp],
             'norm_fp': fp[args.norm_fp], 'ffn_fp': fp[args.ffn_fp], 'pe_fp': fp[args.pe_fp], }

pe_config = {'exp': args.exp, '1d': args.dim, 'imp': args.imp, 'log': args.log,
             'flash_train': args.flash_train, 'flash_test': args.flash_test, 'init': args.init,
             'post': args.post_norm_attn, 'both': args.post_norm_ffn, }

assert args.key != ''
key = args.key
# key = f"{'xpos' if args.exp else 'rope'}_{'inv' if args.imp else 'imp'}_{'1d' if args.dim else '2d'}_" \
#       f"{'log' if args.log else 'raw'}_{str(datetime.now())[2:10].replace('-', '')}" if args.key == '' else args.key
modification = args.modification

setup_seed(42)
torch.set_default_dtype(fp_config['default'])

max_length = 512
model_tag = 'clm_arxiv_1'

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

assert model_tag in model_args and (model_tag, max_length) in train_args

model_args, train_args = model_args[model_tag], train_args[(model_tag, max_length)]
head_dim = model_args['hidden_size'] // model_args['num_attention_heads']

# # todo: https://www.deepspeed.ai/docs/config-json/
#
# ds_config = {
#
#     "fp16": {"enabled": True if fp_config['default'] == torch.float16 else False},
#     "bf16": {"enabled": True if fp_config['default'] == torch.bfloat16 else False},
#
#     "zero_allow_untested_optimizer": True,
#     "zero_force_ds_cpu_optimizer": False,
#
#     "zero_optimization": {
#         "stage": 3,
#         "overlap_comm": True,
#         "contiguous_gradients": True,
#         "sub_group_size": 1e7,
#         "stage3_max_live_parameters": 1e7,
#         "stage3_max_reuse_distance": 1e7,
#         "stage3_gather_16bit_weights_on_model_save": True
#     },
#
#     "gradient_accumulation_steps": 1,
#     "steps_per_print": 2000,
#     "train_batch_size": train_args['per_device_train_batch_size'] * world_size,
#     "wall_clock_breakdown": False
# }
#
# dschf = HfDeepSpeedConfig(ds_config)
#
# # todo: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py#L603
#
# with deepspeed.zero.Init(dtype=fp_config['default'], config_dict_or_path=ds_config):  # bfloat16, float16, float32

config = AutoConfig.from_pretrained('/remote-home/share/llama_hf/7B')
config.gradient_checkpointing = True
config.torch_dtype = fp_config['default']  # bfloat16, float16, float32
config.hidden_size = model_args['hidden_size']                  # 4096
config.intermediate_size = model_args['intermediate_size']      # 11008
config.num_attention_heads = model_args['num_attention_heads']  # 32
config.num_hidden_layers = model_args['num_hidden_layers']      # 32

model = LlamaForCausalLM(config=config, pe_config=pe_config, fp_config=fp_config)  # .bfloat16()

train_args['fp16'] = True if fp_config['default'] == torch.float16 else False
train_args['fp16_full_eval'] = True if fp_config['default'] == torch.float16 else False
train_args['bf16'] = True if fp_config['default'] == torch.bfloat16 else False
train_args['bf16_full_eval'] = True if fp_config['default'] == torch.bfloat16 else False

train_args['output_dir'] = 'checkpoints_fp'

# if rank == 0:
#     import pdb
#     pdb.set_trace()
# torch.distributed.barrier()

if local_rank == 0:
    print('model type :', key, '\n')
    print('pe_config :', pe_config, '\n')
    print('fp_config :', fp_config, '\n')
    print('model_args :', model_args, '\n')
    print('train_args :', train_args, '\n')
    print('modification :', modification, '\n')

tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B', use_fast=False)
tokenizer.pad_token_id = 0

if local_rank == 0:
    print('tokenizer is over !')

dataset_info = get_dataset_info('arxiv')
train_dataset = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.train_split).data
eval_dataset_ = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.test_split, test_note='extra').data

eval_dataset = {}
prefix_list = ['512', '1024', '2048', '3072', '4096', '5120', '6144', '7168', '8192', '9216', '10240', ]  #
# prefix_list = ['128', '512', '1024', '2048', '3072', '4096', '5120', '6144', ]
# '256', '768', '1536', '2560', '3584', '4608', '5632', '6656', '7168',

if local_rank == 0:
    print(prefix_list)

for prefix in prefix_list:
    eval_dataset[prefix] = eval_dataset_[prefix]

if local_rank == 0:
    print('dataset is over !')

# todo：https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

model_path = key  # str(datetime.now())[:10].replace('-', '_') + '-' +
train_args['output_dir'] = '/'.join([train_args['output_dir'], model_path])

if local_rank == 0:
    print('checkpoints and model will be saved in', train_args['output_dir'])

training_args = TrainingArguments(report_to='none', load_best_model_at_end=True,
                                  metric_for_best_model='eval_512_ppl', greater_is_better=False,
                                  **train_args)  # deepspeed=ds_config,

trainer = TrainerForCausalLM(model=model, args=training_args, tokenizer=tokenizer,
                             train_dataset=train_dataset, eval_dataset=eval_dataset,
                             data_collator=DataCollatorForCausalLM(),
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
                             )

trainer.train()

if local_rank == 0:
    print('training is over !')

trainer.save_model(f'{trainer.args.output_dir}/train_last')
