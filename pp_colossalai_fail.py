import argparse
from typing import Callable, List, Union
from datasets import load_dataset
from transformers import AutoTokenizer

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AlbertForSequenceClassification,
    AutoConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator

import torch.distributed as dist
from colossalai.launch import launch_from_torch  # 分布式启动工具

# 在代码最开始添加初始化
def init_distributed():
    try:
        colossalai.launch_from_torch(
            config={},
            seed=42,
        )
    except Exception as e:
        # 如果colossalai初始化失败，使用基础的PyTorch分布式初始化
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )
    
    # 设置当前设备
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

# 在主代码开始处调用初始化
init_distributed()



# Define some config
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

coordinator = DistCoordinator()

# 加载MRPC数据集
dataset = load_dataset('glue', 'mrpc')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['sentence1'], 
                    examples['sentence2'],
                    truncation=True,
                    padding='max_length',
                    max_length=128)

# 处理数据集
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 转换为PyTorch数据集格式
class MRPCDataset(Dataset):
    def __init__(self, encoded_dataset, split='train'):
        self.dataset = encoded_dataset[split]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['label'])
        }

# 创建数据加载器
train_dataset = MRPCDataset(encoded_dataset, 'train')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}

# Define 'criterion' function
def _criterion(outputs, inputs):
    return outputs.loss

# Define Bert model
cfg = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=cfg).cuda()

# Define optimizer (使用PyTorch原生优化器)
lr = LEARNING_RATE
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = Adam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

# Define lr_scheduler
total_steps = len(train_dataloader) * NUM_EPOCHS
num_warmup_steps = int(WARMUP_FRACTION * total_steps)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps,
)

plugin = HybridParallelPlugin(tp_size=1,
                             pp_size=2,
                             num_microbatches=None,
                             microbatch_size=1,
                             enable_all_optimization=True,
                             zero_stage=1,
                             precision='fp16',
                             initial_scale=1)
booster = Booster(plugin=plugin)

# 只boost模型，不boost优化器
model, _, _criterion, _, lr_scheduler = booster.boost(model,
                                                     criterion=_criterion,
                                                     lr_scheduler=lr_scheduler)

# Define a train function
def train_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, _criterion: Callable, lr_scheduler: LRScheduler,
                train_dataloader: DataLoader, booster: Booster, coordinator: DistCoordinator):

    is_pp_last_stage = booster.plugin.stage_manager.is_last_stage()
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    
    with tqdm(range(total_step),
              desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]',
              disable=not (is_pp_last_stage)) as pbar:
        for _ in pbar:
            outputs = booster.execute_pipeline(train_dataloader_iter,
                                             model,
                                             _criterion,
                                             optimizer,
                                             return_loss=True)
            
            if is_pp_last_stage:
                loss = outputs['loss']
                pbar.set_postfix({'loss': loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

# Train model
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)