# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:08

import sys
sys.path.append('..')

import copy
import json
import os
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser
from data_processer import DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, CorpusPreprocess, TokenIds
from aigc_zoo.model_zoo.llm.ppo_model import PetlArguments,LoraConfig,PPOArguments,PPOConfig
from config.rlhf_stage3_ppo_config import *



def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper(DataHelper):
    index = 1

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)

    def load_tokenizer_and_config(self,*args,**kwargs):
        ret = super().load_tokenizer_and_config(*args,**kwargs)
        self._preprocess_tokenizer_config()
        return ret

    def _preprocess_tokenizer_config(self):
        model_args = self.model_args
        tokenizer = self.tokenizer
        config = self.config
        if "llama" in model_args.model_name_or_path.lower() and tokenizer.bos_token_id != DEFAULT_BOS_TOKEN:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
            if tokenizer.pad_token_id is None or tokenizer.pad_token_id == -1:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({
                "pad_token": tokenizer.eos_token,
            })
        if config.decoder_start_token_id is None:
            config.decoder_start_token_id = config.bos_token_id
        assert config.decoder_start_token_id == config.bos_token_id

    def on_get_labels(self, files: typing.List[str]):
        D = ['score']
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label


    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]

        ppo_args:PPOConfig = self.external_kwargs['ppo_args']
        max_new_tokens = ppo_args.gen_kwargs['max_new_tokens']
        tokenizer = self.tokenizer

        pair_data = data
        d = TokenIds.process(pair_data,tokenizer,max_seq_length,max_new_tokens)
        if self.index < 3:
            print(d)
        return d

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            d = CorpusPreprocess.process(lines)
            D.extend(d)
        return D

    def collate_fn(self, batch):
        merge_keys = ['input_ids','attention_mask']
        batch = copy.copy(batch)
        o = {
            k: []
            for k in batch[0].keys()
        }
        for i, b in enumerate(batch):
            for k in b:
                o[k].append(copy.deepcopy(b[k]))

        o_pad = {
            k: o[k] for k in merge_keys
        }
        tokenizer: PreTrainedTokenizer = self.tokenizer
        o_pad = tokenizer.pad(o_pad,
                              # max_length=self.data_args.train_max_seq_length,
                              return_tensors="pt")
        for k in o_pad:
            o[k] = o_pad[k]
        return o


    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = None
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments,PPOArguments))
    model_args, training_args, data_args, lora_args,ppo_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    ppo_args = ppo_args.config

    dataHelper = NN_DataHelper(model_args, training_args, data_args,ppo_args=ppo_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()
    


    # 缓存数据集
    dataHelper.make_dataset_all()


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
