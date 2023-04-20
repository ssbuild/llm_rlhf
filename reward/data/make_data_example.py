# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 9:10
# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50
from datasets import load_dataset
import json
ds = load_dataset("openai/summarize_from_feedback", name="comparisons")

# ds['train'].to_json('./train.json')
# ds['validation'].to_json('./eval.json')


with open('./train.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for d in ds['train']:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

with open('./eval.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for d in ds['validation']:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
