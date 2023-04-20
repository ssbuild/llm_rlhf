## 状态测试中...


## 1.数据示例
    给出一种 生成数据打分格式
    {
     "info": {
       "id": "t3_2vwp1w",
       "post": "I had a car accident on friday, other party involved was speeding and hit me. but because he denies it it seems like I was wrong because he was supposed to go first under normal circumstances. ( give way road markings ) \n\nbut because it was clear when I checked it I drove on, and when I was almost past the intersection he slammed me in the side near the back seat. and caused me to slide across the road for 2-3 meters hit a street light and then bounce back a meter. both doors completely jammed so i had to climb out the window...\n\ncan I somehow get an investigation going about this to see how fast he had to be driving to get this much force in the collision?\nbecause the damage on my car would suggest that he was driving way faster than the legal limit there. ( which is 50 km/h )\n\nalso another reason why i think he was going way faster than admitted is because he could never have reached the intersection from such a distance as where i could not even see him yet\n\n(pictures of the damage:  ) as you can see with the damage, I am lucky to be alive and unharmed right now... 1ft further forward and it could have been my end...\n\nhelp would be appeciated on this :)",
       "title": "Anybody with knowledge of the Dutch law around ? car accident questions.",
       "subreddit": "legaladvice"
     },
     "summaries": [
       {
         "text": " car accident caused me 2-3m damage to my car both doors totally jammed and driving way faster than usual. need info on what to do with this.. thanks :)",
         "policy": "sup4_ppo_rm3_kl10",
         "note": "Was the accident caused by driving fast."
       },
       {
         "text": " we suspect other party involved of speeding when he hit me but I can't prove it without an investigation into the damage, how can i get such an investigation ? if at all possible.",
         "policy": "ref",
         "note": "Unclear what happened."
       }
     ],
     "choice": 1
   }




## 6.生成训练record
   
- cd data && make_data_example.py 
- python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)


## 7.推理
    # infer.py 推理预训练模型
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
     python infer.py



## 8.训练
    支持3种微调方式 
- 冻结 N 层 修改models.py global_num_layers_freeze
- lora
- 正常微调
```text
    python train.py
```

```text
多机多卡训练 例子 3个机器 每个机器 4个卡
修改train.py Trainer num_nodes = 3
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py 
```


### 是否开启lora finetuning
    with_lora


### int高效训练方式
   lora int8     修改modes.py load_in_8bit = True 修改data_utils.py 对应with_lora=True 
   注意 lora int8 多卡训练  CUDA_VISIBLE_DEVICES=显卡数量 和 data_utils.py devices 保持一致，否则容易出故障 ， 建议使用 CUDA_VISIBLE_DEVICES=显卡数量 python train.py

## 9.是否开启deepspeed
    启动则将data_utils.py  修改 enable_deepspeed 
    lora 模式暂时不支持deepspeed
