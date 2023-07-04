"""
  - BERT_base (L=12, H=768, A=12, #Para=110M) and BERT_large (L=24, H=1024, A=16, #Para=340M)
  - FNet hyper L=2, H=128, A=2; L=2, H=256, A=4; L=4 , H=256, A=4; L=4 , H=512, A= 8;
               L=8, H=256, A=4; L=8, H=512, A=8; L=12, H=512, A=8; L=12, H=768, A=12;
  - batch size: 16, 32; Learning rate (Adam): 5e-5, 3e-5, 2e-5; Number of epochs: 2, 3, 4
"""

model_args = {
    'clm_arxiv_0': {
        'hidden_size': 896, 'intermediate_size': 3584, 'num_attention_heads': 12, 'num_hidden_layers': 16,
    },
    'clm_arxiv_1': {
        'hidden_size': 1024, 'intermediate_size': 4096, 'num_attention_heads': 16, 'num_hidden_layers': 16,
    },
    'clm_arxiv_2': {
        'hidden_size': 1024, 'intermediate_size': 4096, 'num_attention_heads': 8, 'num_hidden_layers': 16,
    },
    'clm_arxiv_8192': {
        'hidden_size': 1024, 'intermediate_size': 4096, 'num_attention_heads': 16, 'num_hidden_layers': 16,
    },
    'clm_llama_f': {
        'hidden_size': 4096, 'intermediate_size': 11008, 'num_attention_heads': 32, 'num_hidden_layers': 32,
    },
}

train_args = {
    ('clm_arxiv_1', 512): {
        'per_device_train_batch_size': 12, 'per_device_eval_batch_size': 12, 'num_train_epochs': 1,  # 4 cards
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.00015, 'weight_decay': 0.1, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'steps', 'eval_steps': 600, 'save_strategy': 'steps', 'save_steps': 600,
        'logging_strategy': 'steps', 'logging_steps': 10, 'eval_accumulation_steps': 1, 'max_grad_norm': 1.0,
        'output_dir': 'checkpoints',
    },
}
