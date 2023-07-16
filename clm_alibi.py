import torch

from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments

from utils.base import setup_seed
from utils.clm_tools import MyDataset, MyDataloader, get_dataset_info, DataCollatorForCausalLM
from utils.clm_trainer import TrainerForCausalLM
from configs.clm_train_config import model_args, train_args

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import EarlyStoppingCallback

from models.llama_with_alibi import LlamaForCausalLM

import sys

setup_seed(42)
torch.set_default_dtype(torch.bfloat16)

model_tag = 'clm_arxiv_1'

max_length, world_size, modification = sys.argv[2], sys.argv[3], '' if len(sys.argv) <= 3 else sys.argv[4]
max_length = int(max_length)

assert model_tag in model_args and (model_tag, max_length) in train_args

model_args, train_args = model_args[model_tag], train_args[(model_tag, max_length)]
head_dim = model_args['hidden_size'] // model_args['num_attention_heads']

# todo: https://www.deepspeed.ai/docs/config-json/

ds_config = {
    "bf16": {
        "enabled": True
    },

    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,

    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e7,
        "stage3_max_live_parameters": 1e7,
        "stage3_max_reuse_distance": 1e7,
        "stage3_gather_16bit_weights_on_model_save": True
    },

    "gradient_accumulation_steps": 1,
    "steps_per_print": 2000,
    "train_batch_size": train_args['per_device_train_batch_size'] * world_size,
    "wall_clock_breakdown": False
}

dschf = HfDeepSpeedConfig(ds_config)

# todo: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py#L603

with deepspeed.zero.Init(dtype=torch.bfloat16, config_dict_or_path=ds_config):

    config = AutoConfig.from_pretrained('/remote-home/share/llama_hf/7B')
    config.gradient_checkpointing = True
    config.torch_dtype = torch.bfloat16
    config.hidden_size = model_args['hidden_size']                  # 4096
    config.intermediate_size = model_args['intermediate_size']      # 11008
    config.num_attention_heads = model_args['num_attention_heads']  # 32
    config.num_hidden_layers = model_args['num_hidden_layers']      # 32

    model = LlamaForCausalLM(config=config)  # .bfloat16()

rank = torch.distributed.get_rank()

# if rank == 0:
#     import pdb
#     pdb.set_trace()
# torch.distributed.barrier()

if rank == 0:
    print('model type is alibi , max length is', max_length, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print('model is over !', '\n')
    print('modification :', modification, '\n')

tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B', use_fast=False)
tokenizer.pad_token_id = 0

if rank == 0:
    print('tokenizer is over !')

dataset_info = get_dataset_info('arxiv')
train_dataset = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.train_split).data
eval_dataset_ = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.test_split, test_note='extra').data

eval_dataset = {}
prefix_list = ['512', '1024', '2048', '3072', '4096', '5120', '6144', '7168', '8192', '9216', '10240', ]  #
# '256', '768', '1536', '2560', '3584', '4608', '5632', '6656', '7168',

if rank == 0:
    print(prefix_list)

for prefix in prefix_list:
    eval_dataset[prefix] = eval_dataset_[prefix]

if rank == 0:
    print('dataset is over !')

# todo：https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

model_path = f'alibi-{max_length}'
train_args['output_dir'] = '/'.join([train_args['output_dir'], model_path])

if rank == 0:
    print('checkpoints and model will be saved in', train_args['output_dir'])

training_args = TrainingArguments(deepspeed=ds_config, report_to='none', load_best_model_at_end=True,
                                  metric_for_best_model=f'eval_{max_length}_ppl', greater_is_better=False,
                                  **train_args)

trainer = TrainerForCausalLM(model=model, args=training_args, tokenizer=tokenizer,
                             train_dataset=train_dataset, eval_dataset=eval_dataset,
                             data_collator=DataCollatorForCausalLM(),
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
                             )

trainer.train()

if rank == 0:
    print('training is over !')

trainer.save_model(f'{trainer.args.output_dir}/train_last')
