# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:09

import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class CorpusPreprocess:
   
    @classmethod
    def process(cls,lines):
        D = []
        for i, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            prompt = jd['prompt']
            # response = jd['response']
            chosen = jd['chosen']
            rejected = jd['rejected']
            if chosen == rejected:
                print('warning text_a == text_b and it will be ingored')
                continue
            D.append((prompt, chosen))
        return D


class TokenIds:


    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,max_seq_length: int,max_new_tokens: int):
        prompt, labels = pair_data

        max_prompt_length = max_seq_length - max_new_tokens

        o = tokenizer(prompt, truncation=True,padding=False, max_length=max_prompt_length)
        input_ids = np.asarray(o['input_ids'],dtype=np.int32)
        attention_mask = np.asarray(o['attention_mask'],dtype=np.int32)

        return {
            "prompt": np.array(bytes(prompt,encoding='utf-8')),
            "org_labels": np.array(bytes(labels, encoding='utf-8')),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }