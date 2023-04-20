# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 9:10

from datasets import load_dataset
import json
ds = load_dataset("CarperAI/openai_summarize_tldr")

with open('./train.json',mode='w',encoding='utf-8',newline='\n') as f:
    for d in ds['train']:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

with open('./eval.json',mode='w',encoding='utf-8',newline='\n') as f:
    for d in ds['valid']:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

with open('./test.json',mode='w',encoding='utf-8',newline='\n') as f:
    for d in ds['test']:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
