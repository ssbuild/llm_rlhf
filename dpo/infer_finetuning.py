# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 11:36
import sys

from aigc_zoo.utils.llm_generate import Generate

sys.path.append('..')
from config.dpo_config import get_deepspeed_config
import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser,AutoConfig,PreTrainedTokenizer
from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.llm.llm_model import MyTransformer

deep_config = get_deepspeed_config()

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, ))
    (model_args, ) = parser.parse_dict(train_info_args,allow_extra_keys=True)

    tokenizer : PreTrainedTokenizer
    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    ckpt_dir = './best_ckpt'
    config = AutoConfig.from_pretrained(ckpt_dir)
    pl_model = MyTransformer(config=config, model_args=model_args)
    if deep_config is None:
        train_weight = './best_ckpt/best.pt'
    else:
        # 建议直接使用转换脚本命令 支持 deepspeed stage 0,1,2,3， 生成 ./best_ckpt/last.ckpt/best.pt 权重文件
        # cd best_ckpt/last.ckpt
        # python zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last.ckpt/best.pt'

    pl_model.load_sft_weight(train_weight)


    pl_model.eval().half().cuda()

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