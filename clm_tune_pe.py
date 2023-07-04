import torch

from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments

from utils.base import setup_seed
from utils.clm_tools import MyDataset, MyDataloader, get_dataset_info, DataCollatorForCausalLM
from utils.clm_trainer import TrainerForCausalLM
from configs.clm_tune_config import model_args, train_args

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.deepspeed import deepspeed_init
from transformers import EarlyStoppingCallback

from models.llama_with_pe import LlamaForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

import sys

setup_seed(42)
torch.set_default_dtype(torch.bfloat16)

max_length = 512
model_tag = 'clm_arxiv_1'

key = sys.argv[2]

assert model_tag in model_args and (model_tag, max_length) in train_args

model_args, train_args = model_args[model_tag], train_args[(model_tag, max_length)]

head_dim = model_args['hidden_size'] // model_args['num_attention_heads']

pe_config = {'1d': key.__contains__('1d'),
             'exp': key.__contains__('xpos'),
             'imp': key.__contains__('imp'),
             'log': True,  # key.__contains__('log'),
             'flash_train': False,
             # (32 < head_dim and key.__contains__('1d')) or (64 < head_dim and key.__contains__('2d')),
             'flash_test': True,
             # (head_dim <= 64 and key.__contains__('1d')) or (head_dim <= 128 and key.__contains__('2d')),
             'post': key.__contains__('post'), 'both': key.__contains__('both'),
             'init': key.__contains__('init'), 'base': 512,
             }

model_path = f'/remote-home/xrliu/projects/FEPE-deepspeed/checkpoints/{key}/train_last/pytorch_model.bin'

config = AutoConfig.from_pretrained('/remote-home/share/llama_hf/7B')
config.gradient_checkpointing = True
config.torch_dtype = torch.bfloat16
config.hidden_size = model_args['hidden_size']  # 4096
config.intermediate_size = model_args['intermediate_size']  # 11008
config.num_attention_heads = model_args['num_attention_heads']  # 32
config.num_hidden_layers = model_args['num_hidden_layers']  # 32

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
    "train_batch_size": 48,
    "wall_clock_breakdown": False
}

dschf = HfDeepSpeedConfig(ds_config)

model = LlamaForCausalLM(config=config, pe_config=pe_config)  # .float().bfloat16()

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "up_proj", "gate_proj", "down_proj"],
                         bias="none", task_type=TaskType.CAUSAL_LM)

model = get_peft_model(model, peft_config)

deepspeed.zero.Init(dtype=torch.bfloat16, config_dict_or_path=ds_config)

rank = torch.distributed.get_rank()
size = torch.distributed.get_world_size()

# train_args['per_device_train_batch_size'] = 48 // size
train_args['per_device_eval_batch_size'] = 48 // size

if rank == 0:
    print('model type is', key, '\n')
    print(pe_config, '\n')
    print(peft_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print('model is over !', '\n')
    model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B', use_fast=False)
tokenizer.pad_token_id = 0

if rank == 0:
    print('tokenizer is over !')

dataset_info = get_dataset_info('arxiv')
train_dataset = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.train_split).data
eval_dataset_ = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.test_split).data

eval_dataset = {}
prefix_list = ['128', '512', '1024', '2048', '3072', '4096', '5120', '6144', ]

if rank == 0:
    print(prefix_list)

for prefix in prefix_list:
    eval_dataset[prefix] = eval_dataset_[prefix]

if rank == 0:
    print('dataset is over !')

model_path = key + '-' + 'lora_log'
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
