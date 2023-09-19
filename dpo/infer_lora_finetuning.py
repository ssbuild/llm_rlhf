# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 11:36
import sys

from aigc_zoo.utils.llm_generate import Generate

sys.path.append('..')

import os
import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser,AutoConfig,PreTrainedTokenizer
from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.llm.dpo_model import MyTransformerDPO,PetlArguments

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, ))
    model_args, data_args = parser.parse_dict(train_info_args,allow_extra_keys=True)

    tokenizer : PreTrainedTokenizer
    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    ckpt_dir = './best_ckpt'
    config = AutoConfig.from_pretrained(ckpt_dir)
    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformerDPO(config=config, model_args=model_args, lora_args=lora_args,
                                   torch_dtype=config.torch_dtype,
                                   new_num_tokens=new_num_tokens,
                                   
                                   # # device_map="auto",
                                   # device_map={"": 0},
                                   )
    # 加载sft权重
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False

    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'),merge_lora_weight=True)
    else:

        pl_model.requires_grad_(False)
        model = pl_model.get_llm_model()

        text_list = ["写一个诗歌，关于冬天",
                     "晚上睡不着应该怎么办",
                     "从南京到上海的路线",
                     ]
        for input in text_list:
            response = Generate.generate(model, query=input, tokenizer=tokenizer, max_length=512,
                                         eos_token_id=config.eos_token_id,
                                         do_sample=False, top_p=0.7, temperature=0.95, )
            print('input', input)
            print('output', response)