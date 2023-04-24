# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:09

import json

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



class CorpusPreprocess:
    # {
    #     "prompt": "SUBREDDIT: r/dating_advice\nTITLE: First date ever, going to the beach. Would like some tips\nPOST: Hey Reddit! I (20M) would like some tips, because I have my first ever date tomorrow (although I've had a gf for 3 years, but no actual dating happened), and we're going to the beach.\n\nI met this girl, we have mutual friends, at a festival a few days ago. We didn't kiss, but we talked, held hands, danced a bit. I asked her to go on a date with me, which was super hard as it is the first time I've asked this to anybody. What I mean to say is, it's not like a standard *first* date because we already spent some time together.\n\nI'm really nervous and excited. I'm going to pick her up tomorrow, we're cycling to the beach which will take 30 minutes, and then what? I'm a bit scared. Should I bring something (the weather, although no rain and sunny, is not super so no swimming), should we do something. I'd like all the tips I can get. Thanks!\nTL;DR: ",
    #     "label": "First date after 3 years in a relationship, going to the beach, terrified. What to bring with me, what to do?"
    # }
    @classmethod
    def process(cls,lines):
        D = []
        for i, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            prompt = jd['prompt']
            label = jd['label']
            D.append((prompt, label))
        return D


class TokenIds:

    @classmethod
    def get_prompt(cls,prompt,tokenizer: PreTrainedTokenizer,max_seq_length: int):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        assert max_seq_length > 5
        tmp = tokenizer.decode(
            tokenizer(
                prompt.split("TL;DR:")[0],
                truncation=True,
                max_length=max_seq_length - 5,  # to make sure "TL;DR" dont get truncated
                add_special_tokens=False,
            )["input_ids"],
            skip_special_tokens=True,
        ).strip()
        tmp = tmp + "\nTL;DR:"
        formatted_prompt = tokenizer.decode(
            tokenizer(tmp, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        ).strip()
        return formatted_prompt

    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,max_seq_length: int,max_target_length: int):
        prompt, label = pair_data
        fprompt = cls.get_prompt(prompt,tokenizer,max_seq_length - max_target_length)

        o = tokenizer.encode_plus(fprompt, truncation=True, max_length=max_seq_length)
        input_ids = np.asarray(o['input_ids'],dtype=np.int32)
        attention_mask = np.asarray(o['attention_mask'],dtype=np.int32)

        seqlen = len(input_ids)
        pad_val = tokenizer.pad_token_id
        pad_len = max_seq_length - seqlen
        if pad_len:
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))

        return {
            "prompt": np.array(bytes(fprompt,encoding='utf-8'),dtype=np.bytes),
            "label": np.array(bytes(label, encoding='utf-8'), dtype=np.bytes),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seqlen": np.asarray(seqlen, dtype=np.int32),
        }