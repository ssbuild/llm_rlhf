# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 8:32
import json
import typing

import numpy as np
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



class CorpusPreprocess:
   
    @classmethod
    def process(cls,tokenizer,lines):
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
                continue
            D.append((prompt,chosen, rejected))
        return D



class TokenIds:
    @classmethod
    def trunction_ids(cls,a_ids: typing.List,b_ids: typing.List,max_seq_length,mode='left',a_min_length=10):
        assert a_min_length < max_seq_length
        a_len = max(a_min_length,max_seq_length - len(b_ids))
        ids = a_ids[:a_len] + b_ids if mode == 'left' else a_ids + b_ids
        return ids[:max_seq_length]
    @classmethod
    def get_prompt_length(cls,a_ids,b_ids):
        l = min(len(a_ids), len(b_ids))
        mask: np.ndarray = a_ids[:l] == b_ids[:l]
        if mask.all():
            return l
        return mask.tolist().index(0)

    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,max_seq_length: int):
        prompt, chosen, rejected = pair_data


        a_ids = tokenizer.encode(prompt, truncation=True, max_length=max_seq_length)
        b_ids1 = tokenizer.encode(chosen, truncation=True, max_length=max_seq_length)
        b_ids2 = tokenizer.encode(chosen, truncation=True, max_length=max_seq_length) + [tokenizer.eos_token_id]

        input_ids_a = np.asarray(cls.trunction_ids(a_ids,b_ids1,max_seq_length-1,mode="left",a_min_length=10) + [tokenizer.eos_token_id] ,
                                 dtype=np.int32)
        attention_mask_a = np.asarray([1] * len(input_ids_a),dtype=np.int32)

        input_ids_b = np.asarray(cls.trunction_ids(a_ids,b_ids2,max_seq_length-1,mode="left",a_min_length=10) + [tokenizer.eos_token_id] ,
                                 dtype=np.int32)
        attention_mask_b = np.asarray([1] * len(input_ids_b),dtype=np.int32)

        seqlen_a = len(input_ids_a)
        seqlen_b = len(input_ids_b)

        if seqlen_a == seqlen_b:
            if np.all(input_ids_a == input_ids_b):
                return None
        a_ids = np.asarray(a_ids,dtype=np.int32)
        pos_a = cls.get_prompt_length(a_ids,input_ids_a)
        pos_b = cls.get_prompt_length(a_ids,input_ids_b)
        assert pos_a >= 0 and pos_a < max_seq_length -1 and pos_b >= 0 and pos_b < max_seq_length -1
        labels = np.asarray([-100] * pos_a + input_ids_a[pos_a:].tolist(),dtype=np.int64)
        labels2 = np.asarray([-100] * pos_b + input_ids_b[pos_b:].tolist(),dtype=np.int64)
        return {
            "input_ids": input_ids_a,
            "attention_mask": attention_mask_a,
            "labels": labels,
            # "seqlen": np.asarray(seqlen_a,dtype=np.int32),
            "input_ids2": input_ids_b,
            "attention_mask2": attention_mask_b,
            "labels2": labels2,
            # "seqlen2": np.asarray(seqlen_b, dtype=np.int32),
        }