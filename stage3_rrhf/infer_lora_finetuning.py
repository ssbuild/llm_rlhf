# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 11:36

import os
import numpy as np
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser,AutoConfig,PreTrainedTokenizer
from data_utils import train_info_args, NN_DataHelper,global_args
from aigc_zoo.model_zoo.llm.rrhf_model import MyRRHFTransformer,PetlArguments

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args,allow_extra_keys=True)

    tokenizer : PreTrainedTokenizer
    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    ckpt_dir = './best_ckpt'
    config = AutoConfig.from_pretrained(ckpt_dir)
    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyRRHFTransformer(config=config, model_args=model_args, lora_args=lora_args,
                                 torch_dtype=config.torch_dtype,
                                 new_num_tokens=new_num_tokens,
                                 
                                 # # device_map="auto",
                                 # device_map={"": 0},
                                 )
    # 加载sft权重
    pl_model.load_sft_weight(ckpt_dir)
    if getattr(pl_model.get_llm_model(), "is_loaded_in_8bit", False):
        pl_model.eval().cuda()
    else:
        pl_model.eval().half().cuda()

    enable_merge_weight = False

    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'),merge_lora_weight=True)
    else:

        pl_model.requires_grad_(False)

        input_list = [
            "\n\nHuman:如何培养土豆\n\nAssistant:土豆生长在地下,然后发送的干子称为花生,这些花生成长为我们熟悉的土豆。",
            "\n\nHuman:如何培养土豆\n\nAssistant:土豆在地下生长成大、坚固的花生,一旦土豆长大了,它们就生长在地上。",
            "\n\nHuman:火柴是怎样制造的?\n\nAssistant:我猜你问我如何制造某些东西,但我们以前从未真正讨论过制造的细节。",
            "\n\nHuman:火柴是怎样制造的?\n\nAssistant:对不起,我担心我不明白你的问题。",
        ]
        input_list = [_[:256] for _ in input_list]
        tokend = tokenizer(input_list,padding=True,truncation=True)
        input_ids = torch.tensor(tokend["input_ids"],dtype=torch.int32).to(pl_model.device)
        output = pl_model.backbone.compute_loss(input_ids=input_ids)
        _,scores = output

        for text,score in zip(input_list,scores):
            print('score:' ,score, "text ",text.replace('\n',''))