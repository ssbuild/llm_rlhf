# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 8:32
import json

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
            text_a = prompt + chosen
            text_b = prompt + rejected
            if text_a == text_b:
                continue
            D.append((text_a, text_b))
        return D

class TokenIds:
    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,max_seq_length: int):
        text_a, text_b = pair_data
        assert text_a != text_b , ValueError('data must not be same')

        o1 = tokenizer.encode_plus(text_a, truncation=True, max_length=max_seq_length)
        o2 = tokenizer.encode_plus(text_b, truncation=True, max_length=max_seq_length)

        input_ids_a = np.asarray(o1['input_ids'],dtype=np.int32)
        attention_mask_a = np.asarray(o1['attention_mask'],dtype=np.int32)

        input_ids_b = np.asarray(o2['input_ids'],dtype=np.int32)
        attention_mask_b = np.asarray(o2['attention_mask'],dtype=np.int32)

        seqlen_a = len(input_ids_a)
        seqlen_b = len(input_ids_b)


        if seqlen_a == seqlen_b:
            if np.all(input_ids_a == input_ids_b):
                return None

        return {
            "input_ids": input_ids_a,
            "attention_mask": attention_mask_a,
            # "seqlen": np.asarray(seqlen_a,dtype=np.int32),
            "input_ids2": input_ids_b,
            "attention_mask2": attention_mask_b,
            # "seqlen2": np.asarray(seqlen_b, dtype=np.int32),
        }