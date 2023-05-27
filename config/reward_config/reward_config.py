# coding=utf8
# @Time    : 2023/5/7 17:28
# @Author  : tk
# @FileName: reward_config
import json
import os

import torch
from transformers import BitsAndBytesConfig

# 默认禁用lora 相关模块 , lora 和 adalora 只能同时启用一个
global_args = {
    "load_in_8bit": False, # lora 如果显卡支持int8 可以开启
    "load_in_4bit": False,

    #load_in_4bit 量化配置
    "quantization_config": None,
    "num_layers": -1, # 是否使用骨干网络的全部层数 ， -1 表示全层, 否则只用只用N层
    "num_layers_key":  "num_hidden_layers",
}


train_info_args = {
    'devices': 1,
    'data_backend': 'record',
    'model_type': 'opt',
    # 预训练模型路径 , 从0训练，则置空

    # 'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-125m',
    # 'config_name': '/data/nlp/pre_models/torch/opt/opt-125m/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/opt/opt-125m',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    # 'config_name': '/data/nlp/pre_models/torch/bloom/bloom-560m/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
    # 'config_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7',

    'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-350m',
    'config_name': '/data/nlp/pre_models/torch/opt/opt-350m/config.json',
    'tokenizer_name': '/data/nlp/pre_models/torch/opt/opt-350m',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
    # 'config_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf',

    'convert_onnx': False, # 转换onnx模型
    'do_train': True,
    'train_file':  [ './data/train.json'],
    'max_epochs': 20,
    'max_steps': -1,
    'optimizer': 'lion', # one of [lamb,adamw_hf,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit]

    'scheduler_type': 'CAWR', #one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau]
    'scheduler':{'T_mult': 1,
             'rewarm_epoch_num': 0.5,  # 如果 max_epochs is not None !
             # 'T_0': 50000,    # 如果 max_epochs is None , 设定步数
             'verbose': False},

    # 'scheduler_type': 'CAWR',
    # 'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 0.5, 'verbose': False},


    # 切换scheduler类型
    # 'scheduler_type': 'WarmupCosine',
    # 'scheduler': None,

    # 'scheduler_type': 'ReduceLROnPlateau',
    # 'scheduler': None,

    # 'scheduler_type': 'Step',
    # 'scheduler':{ 'decay_rate': 0.999,'decay_steps': 100,'verbose': True},

    # 'scheduler_type': 'CAWR',
    # 'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': True},

    # 'scheduler_type': 'CAL',
    # 'scheduler': {'rewarm_epoch_num': 2,'verbose': True},


    'optimizer_betas': (0.9, 0.999),
    'train_batch_size': 4,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'learning_rate': 2e-5,  #
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length':  512, #
    'max_target_length': 100,  # 预测最大长度, 保留字段
    'use_fast_tokenizer': False,


}




#配置检查


if global_args['load_in_8bit'] == global_args['load_in_4bit'] and global_args['load_in_8bit'] == True:
    raise Exception('load_in_8bit and load_in_4bit only set one at same time!')

