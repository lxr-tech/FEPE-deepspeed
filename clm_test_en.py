import torch
import torch.nn.functional as f

from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments

from utils.base import setup_seed
from utils.clm_tools import MyDataset, MyDataloader, get_dataset_info, DataCollatorForCausalLM
from utils.clm_trainer import TrainerForCausalLM
from configs.clm_base_en_config import model_args, train_args

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.deepspeed import deepspeed_init

from models.llama_with_pe import LlamaForCausalLM

import sys

checkpoints = {
    'rope_inv_2d_raw': ('', ''),
    'rope_inv_2d_log': ('', ''),
    'rope_inv_1d_raw': ('', ''),
    'rope_inv_1d_log': ('', ''),

    'xpos_inv_2d_raw': ('', ''),
    'xpos_inv_2d_log': ('', ''),
    'xpos_inv_1d_raw': ('', ''),
    'xpos_inv_1d_log': ('', ''),

    'rope_imp_2d_raw': ('', ''),
    'rope_imp_2d_log': ('', ''),
    'rope_imp_1d_raw': ('', ''),
    'rope_imp_1d_log': ('', ''),

    'xpos_imp_2d_raw': ('', ''),
    'xpos_imp_2d_log': ('', ''),
    'xpos_imp_1d_raw': ('', ''),
    'xpos_imp_1d_log': ('', ''),
}

setup_seed(42)
torch.set_default_dtype(torch.bfloat16)

max_length = 512
model_tag = 'clm_arxiv_1'

key = sys.argv[2]

assert model_tag in model_args and (model_tag, max_length) in train_args

model_args, train_args = model_args[model_tag], train_args[(model_tag, max_length)]

head_dim = model_args['hidden_size'] // model_args['num_attention_heads']

pe_config = {'exp': key.__contains__('xpos'), '1d': key.__contains__('1d'),
             'imp': key.__contains__('imp'), 'log': key.__contains__('log'),
             'flash_train': False,
             # (32 < head_dim and key.__contains__('1d')) or (64 < head_dim and key.__contains__('2d')),
             'flash_test': True,
             # (head_dim <= 64 and key.__contains__('1d')) or (head_dim <= 128 and key.__contains__('2d')),
             'post': key.__contains__('post'), 'init': key.__contains__('init'), }  # post_norm for attn only

model_path = f'/remote-home/xrliu/projects/FEPE-deepspeed/checkpoints/{key}/train_last/pytorch_model.bin'

config = AutoConfig.from_pretrained('/remote-home/share/llama_hf/7B')
config.gradient_checkpointing = True
config.torch_dtype = torch.bfloat16
config.hidden_size = model_args['hidden_size']  # 4096
config.intermediate_size = model_args['intermediate_size']  # 11008
config.num_attention_heads = model_args['num_attention_heads']  # 32
config.num_hidden_layers = model_args['num_hidden_layers']  # 32

model = LlamaForCausalLM(config=config, pe_config=pe_config)  # .float().bfloat16()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

ds_config = 'ds_config.json'
dschf = HfDeepSpeedConfig(ds_config)

deepspeed.zero.Init(dtype=torch.bfloat16, config_dict_or_path=ds_config)

rank = torch.distributed.get_rank()
size = torch.distributed.get_world_size()

train_args['per_device_train_batch_size'] = 48 // size
train_args['per_device_eval_batch_size'] = 48 // size

if rank == 0:
    print('model type is', key, '\n')
    print(pe_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print('model is over !', '\n')

tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B', use_fast=False)
tokenizer.pad_token_id = 0

if rank == 0:
    print('tokenizer is over !')

dataset_info = get_dataset_info('arxiv')
train_dataset = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.train_split).data
eval_dataset_ = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.test_split, test_note='extra').data  #

eval_datasets = {}
prefix_list = ['512', '1024', '2048', '3072', '4096', '5120', '6144', '7168', '8192', '9216', '10240', ]  #
# prefix_list = ['128', '512', '1024', '2048', '3072', '4096', '5120', '6144', ]  #
# 10240, 9216, 8192, 7168, 6144, 4096, 2048, 1024, 512

if rank == 0:
    print(prefix_list)

for prefix in prefix_list:
    eval_datasets[prefix] = eval_dataset_[prefix]

if rank == 0:
    print('dataset is over !')

training_args = TrainingArguments(deepspeed=ds_config, report_to='none', **train_args)

trainer = TrainerForCausalLM(model=model, args=training_args, tokenizer=tokenizer,
                             train_dataset=train_dataset, eval_dataset=eval_datasets,
                             data_collator=DataCollatorForCausalLM(), )

deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(trainer, num_training_steps=trainer.args.max_steps)
trainer.model = deepspeed_engine.module
trainer.model_wrapped = deepspeed_engine
trainer.deepspeed = deepspeed_engine
trainer.optimizer = optimizer
trainer.lr_scheduler = lr_scheduler
trainer.model_wrapped = trainer._wrap_model(trainer.model_wrapped)

# print('args', trainer.args)
# print('ds config', training_args.deepspeed)
# trainer._load_from_checkpoint(model_path)

# if rank == 0:
#     import pdb
#     pdb.set_trace()
# torch.distributed.barrier()
#

if rank == 0:
    print(f'\'{key}\'\n')

if isinstance(trainer.eval_dataset, dict):
    metrics = {}
    for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
        dataset_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=None,
            metric_key_prefix=f"eval_{eval_dataset_name}",
        )
        metrics.update(dataset_metrics)
else:
    metrics = trainer.evaluate(ignore_keys=None)

"""
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as f

from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments

from utils.base import setup_seed
from utils.clm_tools import MyDataset, MyDataloader, get_dataset_info, DataCollatorForCausalLM
from utils.clm_trainer import TrainerForCausalLM
from configs.clm_base_en_config import model_args, train_args

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.deepspeed import deepspeed_init

from models.llama_with_pe import LlamaForCausalLM

import sys

checkpoints = {
    'rope_inv_2d_raw': ('ds_clm_rope_arxiv_1_512', 'checkpoints-230531', 'checkpoint-16920'),  # 8460
    'rope_inv_2d_log': ('ds_clm_rope_arxiv_1l_512', 'checkpoints-230531', 'checkpoint-16920'),  # 12690
    'rope_inv_1d_raw': ('ds_clm_rope_arxiv_1d_512', 'checkpoints-230531', 'checkpoint-16920'),  # 8460
    'rope_inv_1d_log': ('ds_clm_rope_arxiv_1dl_512', 'checkpoints-230531', 'checkpoint-16920'),  # (21150)

    'rope_imp_2d_raw': ('ds_clm_fepe_arxiv_1q_512', 'checkpoints', 'checkpoint-16920'),  # 8460
    'rope_imp_2d_log': ('ds_clm_fepe_arxiv_1t_512', ),
    'rope_imp_1d_raw': ('ds_clm_fepe_arxiv_1n_512', 'checkpoint-16920'),  # 12690
    'rope_imp_1d_log': ('ds_clm_fepe_arxiv_1m_512', 'checkpoint-16920'),  # (21150)

    'xpos_inv_2d_raw': ('2023_06_03_01_02', 'checkpoints', 'checkpoint-16920'),  # 12690
    'xpos_inv_2d_log': ('ds_clm_xpos_arxiv_1l_512', 'checkpoints-230531', 'checkpoint-16920'),  # 16920
    'xpos_inv_1d_raw': ('ds_clm_xpos_arxiv_1d_512', 'checkpoints-230531', 'checkpoint-16920'),  # 16920
    'xpos_inv_1d_log': ('ds_clm_xpos_arxiv_1dl_512', 'checkpoints-230531', 'checkpoint-12690'),  # 16920

    'xpos_imp_2d_raw': ('ds_clm_fepe_arxiv_1o_512', 'checkpoints', 'checkpoint-16920'),
    'xpos_imp_2d_log': ('ds_clm_fepe_arxiv_1p_512', 'checkpoints', 'checkpoint-16920'),  # (21150)
    'xpos_imp_1d_raw': ('ds_clm_fepe_arxiv_1j_512', 'checkpoints', 'checkpoint-12690'),
    'xpos_imp_1d_log': ('ds_clm_fepe_arxiv_1l_512', 'checkpoints', 'checkpoint-12690'),
}

setup_seed(42)
torch.set_default_dtype(torch.bfloat16)

max_length = 512
model_tag = 'clm_arxiv_1'

assert model_tag in model_args and (model_tag, max_length) in train_args

model_args, train_args = model_args[model_tag], train_args[(model_tag, max_length)]

key = sys.argv[-1]

pe_config = {'exp': key.__contains__('xpos'), '1d': key.__contains__('1d'),
             'imp': key.__contains__('imp'), 'log': key.__contains__('log')}
folder1, folder2, folder3 = checkpoints[key]
model_path = f'/remote-home/xrliu/projects/FEPE-deepspeed/{folder2}/{folder1}/{folder3}/'
# model_path = '/remote-home/xrliu/projects/FEPE-deepspeed/checkpoints/ds_clm_fepe_arxiv_1l_512/checkpoint-16920'

model = LlamaForCausalLM.from_pretrained(model_path, pe_config=pe_config)  # .float().bfloat16()

ds_config = 'ds_config.json'
dschf = HfDeepSpeedConfig(ds_config)

with deepspeed.zero.Init(dtype=torch.bfloat16, config_dict_or_path=ds_config):

    config = AutoConfig.from_pretrained('/remote-home/share/llama_hf/7B')
    config.gradient_checkpointing = True
    config.torch_dtype = torch.bfloat16
    config.hidden_size = model_args['hidden_size']                  # 4096
    config.intermediate_size = model_args['intermediate_size']      # 11008
    config.num_attention_heads = model_args['num_attention_heads']  # 32
    config.num_hidden_layers = model_args['num_hidden_layers']      # 32

rank = torch.distributed.get_rank()
size = torch.distributed.get_world_size()

train_args['per_device_train_batch_size'] = 48 // size
train_args['per_device_eval_batch_size'] = 48 // size

if rank == 0:
    print('model type is', key, '\n')
    print(pe_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print('model is over !', '\n')

tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B', use_fast=False)
tokenizer.pad_token_id = 0

if rank == 0:
    print('tokenizer is over !')

dataset_info = get_dataset_info('arxiv')
train_dataset = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.train_split).data
eval_dataset_ = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.test_split, test_note='extra').data  #

eval_datasets = {}
prefix_list = ['512', '1024', '6144', '7168', '8192', '9216', ]  # , '10240'
# prefix_list = ['128', '512', '1024', '2048', '3072', '4096', '5120', '6144', ]  # '2048', '4096',
# 10240, 9216, 8192, 7168, 6144, 4096, 2048, 1024, 512

if rank == 0:
    print(prefix_list)

for prefix in prefix_list:
    eval_datasets[prefix] = eval_dataset_[prefix]

if rank == 0:
    print('dataset is over !')

training_args = TrainingArguments(deepspeed=ds_config, report_to='none', **train_args)

trainer = TrainerForCausalLM(model=model, args=training_args, tokenizer=tokenizer,
                             train_dataset=train_dataset, eval_dataset=eval_datasets,
                             data_collator=DataCollatorForCausalLM(), )

deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                trainer, num_training_steps=trainer.args.max_steps, resume_from_checkpoint=model_path
            )
trainer.model = deepspeed_engine.module
trainer.model_wrapped = deepspeed_engine
trainer.deepspeed = deepspeed_engine
trainer.optimizer = optimizer
trainer.lr_scheduler = lr_scheduler
trainer.model_wrapped = trainer._wrap_model(trainer.model_wrapped)

# print('args', trainer.args)
# print('ds config', training_args.deepspeed)
# trainer._load_from_checkpoint(model_path)

# if rank == 0:
#     import pdb
#     pdb.set_trace()
# torch.distributed.barrier()
#

if rank == 0:
    print('\n')

if isinstance(trainer.eval_dataset, dict):
    metrics = {}
    for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
        dataset_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=None,
            metric_key_prefix=f"eval_{eval_dataset_name}",
        )
        metrics.update(dataset_metrics)
else:
    metrics = trainer.evaluate(ignore_keys=None)
"""