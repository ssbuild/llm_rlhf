# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 8:32
import json
import numpy as np
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


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

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class TokenIds:
    @classmethod
    def process(cls,data,tokenizer: PreTrainedTokenizer,max_seq_length: int):
        query, responses,scores = data
        ds = []
        query_input_ids = tokenizer.encode(query,padding="longest",max_length=max_seq_length,truncation=True)
        for res in responses:
            if stop_response:
                r = stop_response(res)
            else:
                r = res
            res_input_ids = tokenizer.encode(r + tokenizer.eos_token, padding="longest", max_length=max_seq_length, truncation=True)
            _truncate_seq_pair(query_input_ids,res_input_ids,max_seq_length)

            input_ids = query_input_ids + res_input_ids
            labels = [-100] * len(query_input_ids)

            ds.append({
                "input_ids": np.asarray(input_ids,dtype=np.int32),
                "labels": np.asarray(labels,dtype=np.int32),
            })
        return ds