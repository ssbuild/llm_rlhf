# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 8:32
import json
import numpy as np
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



def _single_tokenize(text, tokenizer, max_len=None):
    if max_len is None:
        max_len = tokenizer.model_max_length
    input_ids = tokenizer.encode(
            text,
            # return_tensors="pt",
            padding="longest",
            max_length=max_len,
            truncation=True,
        )
    return input_ids

IGNORE_INDEX = -100
def stop_response(res):
    stops = ['\n\nHuman:', '\n\nAssistant:', '\n\nhuman:', '\n\nassistant:']
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[:res.find(stop)].strip()
    return res


class CorpusPreprocess:
    @classmethod
    def process(cls,tokenizer,lines):
        D = []
        for i, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            prompt = jd['prompt']
            responses = jd['responses']
            scores = jd['scores']
            D.append((prompt, responses, scores))
        return D


class TokenIds:
    @classmethod
    def process(cls,data,tokenizer: PreTrainedTokenizer,max_seq_length: int):
        query, responses,scores = data
        ds = []
        query_input_ids = tokenizer.encode(query,padding="longest",max_length=max_seq_length,truncation=True)
        query_target = np.asarray([IGNORE_INDEX] * (query_input_ids.shape[0] - 1),dtype=np.int32)
        dummy_target = np.asarray([IGNORE_INDEX],dtype=np.int32)
        for res in responses:
            if stop_response:
                r = stop_response(res)
            else:
                r = res

            tokenizer.encode(r + tokenizer.eos_token, tokenizer, padding="longest", max_length=max_seq_length, truncation=True)
            res_input_ids = _single_tokenize(r + tokenizer.eos_token, tokenizer,
                                             max_len=tokenizer.model_max_length - query_input_ids.shape[
                                                 0])  # eos here
            input_ids = np.cat((query_input_ids, res_input_ids), ax=0)
            labels = np.cat((query_target, res_input_ids, dummy_target), dim=0)
            ds.append({
                "input_ids": input_ids,
                "labels": labels,
            })
        return ds