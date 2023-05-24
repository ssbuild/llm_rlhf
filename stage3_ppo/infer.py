# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_lora_finetuning
import sys
sys.path.append('..')

import os
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser,AutoConfig,PreTrainedTokenizer

from data_utils import train_info_args, NN_DataHelper
from models import MyPPOTransformer, Generate,LoraArguments,PPOArguments

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments,PPOArguments))
    model_args, data_args= parser.parse_dict(train_info_args,allow_extra_keys=True)



    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()




    pl_model = MyPPOTransformer(config=config, model_args=model_args, training_args=training_args)

    pl_model.eval().half().cuda()

    model = pl_model.get_llm_model()

    text = "哪些食物对糖尿病患者有好处?"
    response = Generate.generate(model, query=text, tokenizer=tokenizer, max_length=512,
                                      eos_token_id=config.eos_token_id,
                                      do_sample=True, top_p=0.7, temperature=0.95, )
    print('input', text)
    print('output', response)