# @Time    : 2023/4/19 23:02
# @Author  : tk
# @FileName: data_utils

import copy
import json
import os
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.lora.v2 import LoraArguments,LoraConfig
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

# 默认禁用lora 相关模块 , lora 和 adalora 只能同时启用一个
lora_info_args = {
    'with_lora': False,  # 是否启用lora模块
    'lora_type': 'lora',
    'r': 8,
    'target_modules': ['query_key_value'],  # bloom,gpt_neox
    # 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
    # 'target_modules': ['c_attn'], #gpt2
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'fan_in_fan_out': False,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
}

adalora_info_args = {
    'with_lora': False,  # 是否启用adalora模块
    'lora_type': 'adalora',
    'r': 8,
    'target_modules': ['query_key_value'],  # bloom,gpt_neox
    # 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
    # 'target_modules': ['c_attn'], #gpt2
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'fan_in_fan_out': False,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"

    'target_r':8, # Target Lora matrix dimension.
    'init_r': 12, #Intial Lora matrix dimension.
    'tinit': 0, #The steps of initial warmup.
    'tfinal': 0, #The steps of final warmup.
    'deltaT': 1, #Step interval of rank allocation.
    'beta1': 0.85, #Hyperparameter of EMA.
    'beta2': 0.85, #Hyperparameter of EMA.
    'orth_reg_weight': 0.5, #The orthogonal regularization coefficient.
    'total_step': None, #The total training steps.
    'rank_pattern': None, #The saved rank pattern.
}

train_info_args = {
    'devices': 1,
    'data_backend': 'record',
    'model_type': 'bloom',
    # 预训练模型路径 , 从0训练，则置空
    'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    'config_name': '/data/nlp/pre_models/torch/bloom/bloom-560m/config.json',
    'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
    # 'config_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-350m',
    # 'config_name': '/data/nlp/pre_models/torch/opt/opt-350m/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/opt/opt-350m',

    #llama transformer >= 4.28
    # 'model_name_or_path': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
    # 'config_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf',



    'convert_onnx': False, # 转换onnx模型
    'do_train': True,
    'train_file':  [ './data/finetune_train_examples.json'],
    'max_epochs': 20,
    'max_steps': -1,
    'optimizer': 'lion', # one of adamw,adam,lamb,lion

    'scheduler_type': 'CAWR',
    'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 0.5, 'verbose': False},

    # 'scheduler_type': 'linear',# one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau
    # 'scheduler': None,

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
    'train_batch_size': 2,
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
    'do_lower_case': False,

    ##############  lora模块
    'lora': {**lora_info_args},
    'adalora': {**adalora_info_args},

}






enable_deepspeed = False

def get_deepspeed_config():
    # 是否开启deepspeed
    if not enable_deepspeed:
        return None
    with open('./deepspeed.json', mode='r', encoding='utf-8') as f:
        deepspeed_config = json.loads(f.read())
    return deepspeed_config


def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper(DataHelper):
    index = 1

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)



    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        examples = data


        strategy = data_conf['strategy']
        if strategy == DataStrategy.sup:
            ds = TokenSupervision.process(tokenizer, config=config,  max_seq_length=max_seq_length, examples=examples,**data_conf[strategy])
        elif strategy == DataStrategy.unsup:
            ds = TokenUnSupervision.process(tokenizer, config=config,  max_seq_length=max_seq_length, examples=examples, **data_conf[strategy])
        elif strategy == DataStrategy.sub_rounds:
            ds = TokenSupervisionRounds.process(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                            **data_conf[strategy])
        else:
            raise ValueError('Invalid strategy', strategy)
        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    # {
    #     "info": {
    #         "id": "t3_2vwp1w",
    #         "post": "I had a car accident on friday, other party involved was speeding and hit me. but because he denies it it seems like I was wrong because he was supposed to go first under normal circumstances. ( give way road markings ) \n\nbut because it was clear when I checked it I drove on, and when I was almost past the intersection he slammed me in the side near the back seat. and caused me to slide across the road for 2-3 meters hit a street light and then bounce back a meter. both doors completely jammed so i had to climb out the window...\n\ncan I somehow get an investigation going about this to see how fast he had to be driving to get this much force in the collision?\nbecause the damage on my car would suggest that he was driving way faster than the legal limit there. ( which is 50 km/h )\n\nalso another reason why i think he was going way faster than admitted is because he could never have reached the intersection from such a distance as where i could not even see him yet\n\n(pictures of the damage:  ) as you can see with the damage, I am lucky to be alive and unharmed right now... 1ft further forward and it could have been my end...\n\nhelp would be appeciated on this :)",
    #         "title": "Anybody with knowledge of the Dutch law around ? car accident questions.",
    #         "subreddit": "legaladvice"
    #     },
    #     "summaries": [
    #         {
    #             "text": " car accident caused me 2-3m damage to my car both doors totally jammed and driving way faster than usual. need info on what to do with this.. thanks :)",
    #             "policy": "sup4_ppo_rm3_kl10",
    #             "note": "Was the accident caused by driving fast."
    #         },
    #         {
    #             "text": " we suspect other party involved of speeding when he hit me but I can't prove it without an investigation into the damage, how can i get such an investigation ? if at all possible.",
    #             "policy": "ref",
    #             "note": "Unclear what happened."
    #         }
    #     ],
    #     "choice": 1
    # }
    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        tokenizer = self.tokenizer
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                jd = json.loads(line)
                if not jd:
                    continue
                info = jd['info']
                summaries = jd['summaries']
                choice = jd['choice']
                if len(summaries) != 2 or choice not in (0, 1):
                    raise ValueError(
                        f"There should be two summaries with a choice that's either 0 or 1. Received {len(summaries)} summaries and choice={choice}."
                    )

                original_text_field = "post" if info["post"] is not None else "article"
                text_a = summaries[choice]["text"] + " " + tokenizer.bos_token + " " + info[original_text_field]
                text_b = summaries[0 if choice == 1 else 1]["text"] + " " + tokenizer.bos_token + " " + info[original_text_field]
                D.append((text_a,text_b))
        return D

    def collate_fn(self, batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        maxlen = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :maxlen]
        o['attention_mask'] = o['attention_mask'][:, :maxlen]
        o['labels'] = o['labels'][:, :maxlen].long()
        return o




if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()
    config.decoder_start_token_id = config.bos_token_id


    if "llama" in model_args.model_name_or_path.lower() and tokenizer.bos_token_id != DEFAULT_BOS_TOKEN:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == -1:
            tokenizer.pad_token_id = tokenizer.eos_token_id


    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False,shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, shuffle=False,mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, shuffle=False,mode='test')


    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
