# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 9:10
import json
from datasets import load_dataset

# {
# 	"prompt": "\n\nHuman:你能描述如何清除笔记本计算机屏幕上的指纹和污垢的步骤吗?\n\nAssistant:是的,也许。为了清洁你的屏幕,你首先需要使用微纤维布或软溼布,轻轻地擦拭屏幕表面。接下来,你会想抓住一个软,无毛巾,微纤维清洁布,轻轻地擦回屏幕上去除指纹和污垢。\n\nHuman:我可以将伊索普罗素酒精喷射到布上,并以这种方式清洗它吗?\n\nAssistant:",
# 	"response": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
# 	"chosen": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
# 	"rejected": " 是的,你可以直接将它喷射到布上。"
# }

# x = {
# 	"prompt": "\n\nHuman:你能描述如何清除笔记本计算机屏幕上的指纹和污垢的步骤吗?\n\nAssistant:是的,也许。为了清洁你的屏幕,你首先需要使用微纤维布或软溼布,轻轻地擦拭屏幕表面。接下来,你会想抓住一个软,无毛巾,微纤维清洁布,轻轻地擦回屏幕上去除指纹和污垢。\n\nHuman:我可以将伊索普罗素酒精喷射到布上,并以这种方式清洗它吗?\n\nAssistant:",
# 	"response": [
#       " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
#       " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
#       " 是的,你可以直接将它喷射到布上。"
#     ],
# 	"score": [1.0,-1.0]
# }

ds = load_dataset("Dahoas/rm-static")

with open('./train_score.json', mode='w', encoding='utf-8', newline='\n') as f:
    for d in ds['train']:
        if d["chosen"] == d["rejected"]:
            continue

        o = {
            "prompt": d["prompt"],
            "response": [
                d["response"],  d["rejected"],
            ],
            "score": [-1.0,-2.0]
        }
        f.write(json.dumps(o, ensure_ascii=False) + '\n')

with open('./eval_score.json', mode='w', encoding='utf-8', newline='\n') as f:
    for d in ds['test']:
        if d["chosen"] == d["rejected"]:
            continue

        o = {
            "prompt": d["prompt"],
            "response": [
                d["response"], d["rejected"],
            ],
            "score": [-1.0,-2.0]
        }
        f.write(json.dumps(o, ensure_ascii=False) + '\n')
