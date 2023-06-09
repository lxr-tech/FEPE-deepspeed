import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as f

from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments
# from models.llama_with_unk import LlamaForCausalLM

from utils.base import setup_seed
from utils.clm_tools import MyDataset, MyDataloader, get_dataset_info, DataCollatorForCausalLM
from utils.clm_trainer import TrainerForCausalLM
from configs.clm_base_en_config import model_args, train_args

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig


# todo: https://huggingface.co/docs/transformers/main_classes/deepspeed

setup_seed(42)
torch.set_default_dtype(torch.bfloat16)

max_length = 2048
model_tag = 'clm_arxiv_1'
model_type = 'rope'  # 'fepe', 'rope', 'rope_v', 'xpos', 'xpos_v', 'alibi'

if model_type == 'fepe':
    from models.llama_with_fepe import LlamaForCausalLM
elif model_type == 'rope':
    from models.llama_with_rope import LlamaForCausalLM
elif model_type == 'rope_v':
    from models.llama_with_rope_v import LlamaForCausalLM
elif model_type == 'xpos':
    from models.llama_with_xpos import LlamaForCausalLM
elif model_type == 'xpos_v':
    from models.llama_with_xpos_v import LlamaForCausalLM
elif model_type == 'alibi':
    from models.llama_with_alibi import LlamaForCausalLM
else:
    raise KeyError('only support fepe, rope, xpos and alibi')

assert model_tag in model_args and (model_tag, max_length) in train_args

model_args, train_args = model_args[model_tag], train_args[(model_tag, max_length)]

# todo: https://www.deepspeed.ai/docs/config-json/

ds_config = 'ds_config.json'
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
    print('model type is', model_type)
    if model_type == 'fepe':
        print('modification: 1d ver., cube / bias, no imag part, exp(xpos), llama config, sqrt scale')  # log+
    else:
        print('modification: 2d ver., no imag part, llama config')
    print(model_args, '\n')
    print(train_args, '\n')
    print('model is over !')

tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B')
tokenizer.pad_token_id = 0

if rank == 0:
    print('tokenizer is over !')

dataset_info = get_dataset_info('arxiv')
train_dataset = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.train_split).data
eval_dataset_ = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.test_split).data

eval_dataset = {}
prefix_list = ['128', '512', '1024', '2048', '3072', '4096', '5120', '6144', ]
# '256', '768', '1536', '2560', '3584', '4608', '5632', '6656', '7168',

if rank == 0:
    print(prefix_list)

for prefix in prefix_list:
    eval_dataset[prefix] = eval_dataset_[prefix]

if rank == 0:
    print('dataset is over !')

# todo：https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

model_path = str(datetime.now())[:-10].replace('-', '_').replace(' ', '_').replace(':', '_')
train_args['output_dir'] = '/'.join([train_args['output_dir'], model_path])

if rank == 0:
    print('checkpoints and model will be saved in', train_args['output_dir'])

training_args = TrainingArguments(deepspeed=ds_config, report_to='none', **train_args)

trainer = TrainerForCausalLM(model=model, args=training_args, tokenizer=tokenizer,
                             train_dataset=train_dataset, eval_dataset=eval_dataset,
                             data_collator=DataCollatorForCausalLM(), )

trainer.train()

if rank == 0:
    print('training is over !')

trainer.save_model(f'{trainer.args.output_dir}/train_last')

# training_args = TrainingArguments(per_device_train_batch_size=4, per_device_eval_batch_size=4,
#                                   optim='adamw_hf', learning_rate=0.0001, weight_decay=0,
#                                   num_train_epochs=10, lr_scheduler_type='linear', warmup_ratio=0.1,
#                                   do_train=True, do_eval=True, bf16=True, bf16_full_eval=True,
#                                   evaluation_strategy='epoch', eval_accumulation_steps=1,
#                                   logging_strategy='steps', logging_steps=10,
#                                   deepspeed=ds_config, output_dir='checkpoints', )

"""
***** Running eval_128 *****
eval_128_acc : 0.443337 eval_128_ppl : 21.971549 

***** Running eval_512 *****
eval_512_acc : 0.436256 eval_512_ppl : 21.593414 

***** Running eval_1024 *****
eval_1024_acc : 0.427716 eval_1024_ppl : 22.386326 

***** Running eval_2048 *****
eval_2048_acc : 0.419367 eval_2048_ppl : 22.84397 

***** Running eval_3072 *****
eval_3072_acc : 0.408405 eval_3072_ppl : 23.885784 

***** Running eval_4096 *****
eval_4096_acc : 0.401179 eval_4096_ppl : 24.624005 

***** Running eval_5120 *****
eval_5120_acc : 0.390923 eval_5120_ppl : 26.171258 
"""