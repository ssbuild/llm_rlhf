# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:09

import json
from typing import Union, Iterable, List
import numpy as np
import torch
from deep_training.nlp.rl.ilql.ilql_dataset import DialogMessage
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

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

            D.append((prompt, chosen,rejected))
        return D


def tokenize_dialogue(  # noqa: C901
    dialogue: Union[str, Iterable[str]], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_length=2048
) -> List[DialogMessage]:
    """
    Tokenize sample with the interleaved form of (prompt_1, output_1, prompt_2, output_2...)
    """
    if isinstance(dialogue, str):
        bos_token = tokenizer.bos_token or tokenizer.eos_token
        dialogue = [bos_token, dialogue]
    elif isinstance(dialogue, Iterable):
        if len(dialogue) % 2 != 0:
            raise ValueError("Dialogue must have an even number of phrases, alternating prompt and output")
        dialogue = list(dialogue)

    if not dialogue[-1].endswith(tokenizer.eos_token):
        dialogue[-1] = dialogue[-1] + tokenizer.eos_token

    tokenized = [
        DialogMessage(is_output=i % 2 == 1, tokens=tuple(tokenizer(dialogue[i], add_special_tokens=False).input_ids))
        for i in range(len(dialogue))
    ]

    # flip to truncate from the left
    if tokenizer.truncation_side == "left":
        tokenized = [DialogMessage(is_output=m.is_output, tokens=m.tokens[::-1]) for m in tokenized[::-1]]

    # truncate if necessary
    lengths = [len(t.tokens) for t in tokenized]
    cumsum_lengths = [sum(lengths[:i]) for i in range(len(lengths))]
    truncated = [
        DialogMessage(is_output=t.is_output, tokens=t.tokens[: max(max_length - cl, 0)])
        for t, cl in zip(tokenized, cumsum_lengths)
    ]

    # flip back if was fliped to left truncate
    if tokenizer.truncation_side == "left":
        truncated = [DialogMessage(is_output=m.is_output, tokens=m.tokens[::-1]) for m in truncated[::-1]]

    # remove empty messages
    out = [t for t in truncated if len(t.tokens) > 0]

    if out[0].is_output:
        if sum(map(lambda msg: len(msg.tokens), out)) == max_length:
            if tokenizer.truncation_side == "left":
                out[0].tokens = out[0].tokens[1:]
            else:
                out[-1].tokens = out[-1].tokens[:-1]

        out.insert(0, DialogMessage(False, (tokenizer.bos_token_id,)))
    return out

class TokenIds:
    @classmethod
    def process(cls,data,tokenizer: PreTrainedTokenizer,max_seq_length: int,max_new_tokens: int):
        ds = []
        prompt,chosen,rejected = data
        if tokenizer.encode(chosen) == tokenizer.encode(rejected):
            return None
        for diaglogue in ((prompt, chosen, [1.0]),(prompt, rejected, [-1.0])):
            rewards = diaglogue[-1]
            dialogue = diaglogue[:-1]

            sample = tokenize_dialogue(dialogue,tokenizer,max_seq_length)
            returns = np.asarray(rewards, dtype=np.float32)

            length = 0
            input_ids = np.asarray(sum((s.tokens for s in sample), ()),dtype=np.int32)
            attention_mask = np.ones(len(input_ids), dtype=np.int32)
            actions_ixs = []
            for dm in sample:
                if dm.is_output:
                    actions_ixs.append(np.arange(length - 1, length + len(dm.tokens) - 1))

                length += len(dm.tokens)

            if not actions_ixs:
                continue

            states_ixs = np.hstack((*actions_ixs, np.asarray(length - 1)))
            dones = np.asarray([1] * (len(states_ixs) - 1) + [0], dtype=np.int32)
            actions_ixs = np.hstack(actions_ixs)

            # returns = returns - returns.mean()
            # std_returns = returns.std()
            # if not np.isnan(std_returns):
            #     returns = returns / (std_returns + np.finfo(returns.dtype).eps)
            rewards = [np.zeros(len(actions_ixs))]
            for rs, ret in zip(rewards, returns):
                rs[-1] = ret
            rewards = np.asarray(rewards[0],dtype=np.float32)
            # sample_lengths = np.array([len(input_ids)])
            # output_lengths = np.array([len(actions_ixs)])
            # prompt_lengths = sample_lengths - output_lengths
            ds.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "rewards": rewards,
                "actions_ixs": actions_ixs,
                "states_ixs": states_ixs,
                "dones": dones,
            })
        if not ds:
            return None
        return ds