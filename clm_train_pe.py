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

import sys

setup_seed(42)
torch.set_default_dtype(torch.bfloat16)

max_length = 512
model_tag = 'clm_arxiv_1'

key, world_size, modification = sys.argv[2], sys.argv[3], '' if len(sys.argv) <= 3 else sys.argv[4]
# if key not in keys:
#     raise KeyError(f'{key} is not supported in {list(keys)}')

if key == 'alibi':
    from models.llama_with_alibi import LlamaForCausalLM
elif key.__contains__('rope') or key.__contains__('xpos'):
    from models.llama_with_pe import LlamaForCausalLM
# elif key.__contains__('rope'):
#     from ver.v0_1_0609.llama_with_rope_v import LlamaForCausalLM
# elif key.__contains__('xpos'):
#     from ver.v0_1_0609.llama_with_xpos_v import LlamaForCausalLM
else:
    raise KeyError('only support rope, xpos and alibi')

assert model_tag in model_args and (model_tag, max_length) in train_args

model_args, train_args = model_args[model_tag], train_args[(model_tag, max_length)]
head_dim = model_args['hidden_size'] // model_args['num_attention_heads']

pe_config = {'exp': key.__contains__('xpos'), '1d': key.__contains__('1d'),
             'imp': key.__contains__('imp'), 'log': key.__contains__('log'),
             'flash_train': False,
             # (32 < head_dim and key.__contains__('1d')) or (64 < head_dim and key.__contains__('2d')),
             'flash_test': True,
             # (head_dim <= 64 and key.__contains__('1d')) or (head_dim <= 128 and key.__contains__('2d')),
             'post': key.__contains__('post'), 'both': key.__contains__('both'),
             'init': key.__contains__('init'), }  # post_norm for attn only

# todo: https://www.deepspeed.ai/docs/config-json/

ds_config = 'ds_config.json'
dschf = HfDeepSpeedConfig(ds_config)

# todo: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py#L603

with deepspeed.zero.Init(dtype=torch.bfloat16, config_dict_or_path=ds_config):  # bfloat16, float16, float32

    config = AutoConfig.from_pretrained('/remote-home/share/llama_hf/7B')
    config.gradient_checkpointing = True
    config.torch_dtype = torch.bfloat16  # bfloat16, float16, float32
    config.hidden_size = model_args['hidden_size']                  # 4096
    config.intermediate_size = model_args['intermediate_size']      # 11008
    config.num_attention_heads = model_args['num_attention_heads']  # 32
    config.num_hidden_layers = model_args['num_hidden_layers']      # 32

    model = LlamaForCausalLM(config=config, pe_config=pe_config)  # .bfloat16()

# train_args['bf16'] = False
# train_args['bf16_full_eval'] = False
# train_args['fp16'] = False
# train_args['fp16_full_eval'] = False

rank = torch.distributed.get_rank()

# if rank == 0:
#     import pdb
#     pdb.set_trace()
# torch.distributed.barrier()

if rank == 0:
    print('model type is', key, '\n')
    print(pe_config, '\n')
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
# prefix_list = ['128', '512', '1024', '2048', '3072', '4096', '5120', '6144', ]
# '256', '768', '1536', '2560', '3584', '4608', '5632', '6656', '7168',

if rank == 0:
    print(prefix_list)

for prefix in prefix_list:
    eval_dataset[prefix] = eval_dataset_[prefix]

if rank == 0:
    print('dataset is over !')

# todo：https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

model_path = key  # str(datetime.now())[:10].replace('-', '_') + '-' +
train_args['output_dir'] = '/'.join([train_args['output_dir'], model_path])

if rank == 0:
    print('checkpoints and model will be saved in', train_args['output_dir'])

training_args = TrainingArguments(deepspeed=ds_config, report_to='none', load_best_model_at_end=True,
                                  metric_for_best_model='eval_512_ppl', greater_is_better=False,
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
