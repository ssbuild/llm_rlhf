## rlhf


## 数据示例

```json
{
	"prompt": "\n\nHuman:你能描述如何清除笔记本计算机屏幕上的指纹和污垢的步骤吗?\n\nAssistant:是的,也许。为了清洁你的屏幕,你首先需要使用微纤维布或软溼布,轻轻地擦拭屏幕表面。接下来,你会想抓住一个软,无毛巾,微纤维清洁布,轻轻地擦回屏幕上去除指纹和污垢。\n\nHuman:我可以将伊索普罗素酒精喷射到布上,并以这种方式清洗它吗?\n\nAssistant:",
	"response": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
	"chosen": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
	"rejected": " 是的,你可以直接将它喷射到布上。"
}
```


## 生成训练record
   
- cd data && make_data_example.py 
- python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)



## 训练
- lora
```text
    python train.py
```

### 多机多卡训练
```text
 例子 3个机器 每个机器 4个卡
修改train.py Trainer num_nodes = 3
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py 
```


### lora finetuning
```text
       global_args = {
          "load_in_8bit": False, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
          "num_layers": -1, # 是否使用骨干网络的全部层数 ， -1 表示全层, 否则只用只用N层
          "num_layers_key":  "num_layers",
      }
         lora_info_args = {
               'with_lora': True,  # 是否启用lora模块
               'r': 8,
               'target_modules': ['query_key_value'],
               'target_dtype': None,
               'lora_alpha': 32,
               'lora_dropout': 0.1,
               'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
               'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
        }
```


### int高效训练方式
   lora int8   
```text
       global_args = {
          "load_in_8bit": True, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
          "num_layers": -1, # 是否使用骨干网络的全部层数 ， -1 表示全层, 否则只用只用N层
          "num_layers_key":  "num_layers",
      }
         lora_info_args = {
               'with_lora': True,  # 是否启用lora模块
               'r': 8,
               'target_modules': ['query_key_value'],
               'target_dtype': None,
               'lora_alpha': 32,
               'lora_dropout': 0.1,
               'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
               'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
        }
```

## 是否开启deepspeed
    修改 enable_deepspeed 
    lora 模式暂时不支持deepspeed


## bf16精度训练
    Trainer.precision='bf16',#需 pl_model.float() 并注释掉 Trainer.max_grad_norm

## 混合精度训练
    Trainer.precision='16-mixed',#需 pl_model.float() 并注释掉 Trainer.max_grad_norm

## 半精度训练
    Trainer.precision='16',#需 pl_model.float()