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
    # {
    #     "prompt": "\n\nHuman: I eat too much fast food. What's a way to start cooking at home more?\n\nAssistant: The easiest way to start cooking more at home is to cook a few meals a week and keep those meals varied. Planning meals and having a list ready will make cooking much easier. Shopping once a week will also keep your meals varied. Try cooking a few recipes you've never made before. Maybe you could try some Italian, Mexican, or Asian dishes. Cooking more at home is also a great way to save money. You can buy ingredients in bulk and cook large portions, which will last you several days.\n\nHuman: Tell me about buying in bulk. Will that make the process easier?\n\nAssistant: Buying in bulk will definitely make cooking and planning meals easier. The key to buying in bulk is to only buy what you know you will use. Keeping a well-stocked pantry with a variety of dried and canned goods will make cooking easier. Buying in bulk will save you money and give you greater variety in your meals.\n\nHuman: How does one buy in bulk?\n\nAssistant:",
    #     "response": " To buy in bulk, start by shopping at warehouse stores or specialty stores. Make a list of the items you usually use in meals and the amounts you use. Make a note of the unit prices for those items. You can then compare unit prices at warehouse stores to see if they're lower. If they are, buy in bulk from warehouse stores. Also, check online for even bigger savings. Another way to buy in bulk is to join a grocery co-op, which is a group of people who buy food in bulk and then split the cost.",
    #     "chosen": " To buy in bulk, start by shopping at warehouse stores or specialty stores. Make a list of the items you usually use in meals and the amounts you use. Make a note of the unit prices for those items. You can then compare unit prices at warehouse stores to see if they're lower. If they are, buy in bulk from warehouse stores. Also, check online for even bigger savings. Another way to buy in bulk is to join a grocery co-op, which is a group of people who buy food in bulk and then split the cost.",
    #     "rejected": " Buying in bulk is easy. There are stores in your area that offer bulk food. Organic, bulk foods are also widely available online. You can find organic dried pasta, nuts, baking items, and spices in bulk. When shopping for groceries, buy the largest package or container possible to get the largest savings. Organic, bulk food is more expensive than traditional packaged food, but much cheaper than organic grocery stores. You can get many non-perishable items for half price or less when buying in bulk."
    # }
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

            D.append((prompt, chosen, prompt, rejected,1.0,-1.0))
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
        n = int(len(data) / 3 * 2)
        dialogue = data[:n]
        rewards = data[n:]

        sample = tokenize_dialogue(dialogue,tokenizer,max_seq_length)

        print('*' *30 , len(sample))
        if len(sample) == 1:
            print(dialogue)
            print(rewards)
            print(data)

        rewards = np.asarray(rewards, dtype=np.float32)

        length = 0
        input_ids = np.asarray(sample[0].tokens,dtype=np.int32)
        output_ids = np.asarray(sample[1].tokens, dtype=np.int32)
        attention_mask = np.ones(len(input_ids), dtype=np.int32)
        actions_ixs = []
        for phrase in sample:
            if phrase.is_output:
                length = len(phrase.tokens)
                actions_ixs.append(np.arange(0, length - 1))

        states_ixs = np.hstack((*actions_ixs, np.asarray(length - 1)))
        dones = np.asarray([1] * (len(states_ixs) - 1) + [0], dtype=np.int32)
        actions_ixs = np.hstack(actions_ixs)
        states_ixs = states_ixs

        # sample_lengths = np.asarray([len(input_ids),len(output_ids)])
        # output_lengths =np.asarray([len(output_ids)])
        # prompt_lengths = sample_lengths - output_lengths
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_ids": output_ids,
            "rewards": rewards,
            "actions_ixs": actions_ixs,
            "states_ixs": states_ixs,
            "dones": dones,
        }