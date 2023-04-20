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
    @classmethod
    def process(cls,tokenizer,lines):
        D = []
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
            text_b = summaries[0 if choice == 1 else 1]["text"] + " " + tokenizer.bos_token + " " + info[
                original_text_field]
            D.append((text_a, text_b))
        return D


class TokenIds:
    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,max_seq_length: int):
        text_a, text_b = pair_data
        o1 = tokenizer.encode_plus(text_a, truncation=True, max_length=max_seq_length)
        o2 = tokenizer.encode_plus(text_b, truncation=True, max_length=max_seq_length)

        input_ids_a = np.asarray(o1['input_ids'],dtype=np.int32)
        attention_mask_a = np.asarray(o1['attention_mask'],dtype=np.int32)

        input_ids_b = np.asarray(o2['input_ids'],dtype=np.int32)
        attention_mask_b = np.asarray(o2['attention_mask'],dtype=np.int32)

        seqlen_a = len(input_ids_a)
        seqlen_b = len(input_ids_b)

        pad_val = tokenizer.pad_token_id
        pad_len = max_seq_length - seqlen_a
        if pad_len:
            input_ids_a = np.pad(input_ids_a, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask_a = np.pad(attention_mask_a, (0, pad_len), 'constant', constant_values=(0, 0))

        pad_len = max_seq_length - seqlen_b
        if pad_len:
            input_ids_b = np.pad(input_ids_b, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask_b = np.pad(attention_mask_b, (0, pad_len), 'constant', constant_values=(0, 0))

        return {
            "input_ids": input_ids_a,
            "attention_mask": attention_mask_a,
            "seqlen": np.asarray(seqlen_a,dtype=np.int32),
            "input_ids2": input_ids_b,
            "attention_mask2": attention_mask_b,
            "seqlen2": np.asarray(seqlen_b, dtype=np.int32),
        }