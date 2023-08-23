# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 14:26
import sys
sys.path.append('..')

import json
import torch
from deep_training.data_helper import ModelArguments, DataArguments
from tqdm import tqdm
from transformers import HfArgumentParser,AutoConfig,PreTrainedTokenizer

from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.llm.reward_model import MyRewardTransformer,PetlArguments

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

    pl_model = MyRewardTransformer(config=config, model_args=model_args, lora_args=lora_args,
                                   torch_dtype=config.torch_dtype,
                                   new_num_tokens=new_num_tokens,
                                   
                                   # # device_map="auto",
                                   # device_map={"": 0},
                                   )
    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)
    if getattr(pl_model.get_llm_model(), "is_loaded_in_8bit", False):
        pl_model.eval().cuda()
    else:
        pl_model.eval().half().cuda()
    pl_model.requires_grad_(False)


    with open('./data/eval.json',mode='r',encoding='utf-8') as f:
        lines = f.readlines()
    print('predict........')
    acc = 0
    total = 0
    for line in tqdm(lines,total=len(lines)):
        jd = json.loads(line)
        if not jd:
            continue
        input_list = [
            jd['prompt'][:256] + jd['chosen'][:256],
            jd['prompt'][:256] + jd['rejected'][:256],
        ]
        tokend = tokenizer(input_list,padding=True,truncation=True,max_length=512)
        input_ids = torch.tensor(tokend["input_ids"],dtype=torch.int32).to(pl_model.device)
        output = pl_model.backbone.compute_loss(input_ids=input_ids)
        _,scores = output
        total += 1
        if scores[0] >= scores[1]:
           acc += 1

    print('total {} , acc count {} , acc {}'.format(total,acc, acc / total))