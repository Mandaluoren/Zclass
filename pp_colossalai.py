import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from typing import Callable, List, Union
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from torch.optim import Adam
from colossalai.nn.optimizer import HybridAdam
# from colossalai.utils import get_dataloader

import colossalai
# from colossalai.launch import launch_from_torch  # 分布式启动工具

# Define some config
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1


colossalai.launch_from_torch()

# Define 'criterion' function with two inputs, which will be passed to 'execute_pipeline'.
def _criterion(outputs, inputs):
    return outputs.loss


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
data_files = {
    'train': '/root/chj/gees/GeeSibling/examples/datasets/mrpc/train.jsonl',
    'test': '/root/chj/gees/GeeSibling/examples/datasets/mrpc/test.jsonl',
    'validation': '/root/chj/gees/GeeSibling/examples/datasets/mrpc/validation.jsonl'
}

dataset = load_dataset('json', data_files=data_files)

# 加载 MRPC 数据集
# dataset = load_dataset('glue', 'mrpc')

print("加载数据集完成")

# 加载 BERT 分词器和模型
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("加载模型成功")

# 数据集预处理函数
def preprocess_function(examples):
    """对 MRPC 的句子对进行分词和处理"""
    return tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

# 对训练集进行预处理
encoded_train_dataset = dataset['train'].map(preprocess_function, batched=True)

# 将预处理后的数据转换为 PyTorch Dataset
class MRPCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.input_ids = torch.tensor(dataset['input_ids'])
        self.attention_mask = torch.tensor(dataset['attention_mask'])
        self.labels = torch.tensor(dataset['label'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# 将训练数据包装为 PyTorch Dataset
train_dataset = MRPCDataset(encoded_train_dataset)
print("将数据集转换为Pytorch可用格式")

# 使用 DataLoader 加载数据
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 定义优化器
# optimizer = Adam(model.parameters(), lr=2e-5)         # AttributeError: 'Adam' object has no attribute 'backward'

# Define optimizer
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

optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

# 使用 ColossalAI 的 HybridParallelPlugin
plugin = HybridParallelPlugin(tp_size=1, pp_size=4, microbatch_size=2, num_microbatches=4)  # 开启流水线并行，pp_size=2 表示 2 个阶段
booster = Booster(plugin=plugin)

# 将模型、优化器和数据加载器包装到 Booster 中
# model, optimizer, train_dataloader = booster.boost(model, optimizer, train_dataloader)
# model = booster.boost(model=model)[0]
# model, optimizer, _criterion, _, lr_scheduler = booster.boost(model,
#                                                                 optimizer,
#                                                                 criterion=_criterion,
#                                                                 lr_scheduler=lr_scheduler)

model, optimizer , _criterion, _, _ = booster.boost(model, optimizer=optimizer, criterion=_criterion)

# 定义学习率调度器
# scheduler = LinearWarmupLR(optimizer, warmup_steps=100, total_steps=len(train_dataloader) * 3)

# 获取分布式训练日志记录器
logger = get_dist_logger()

# 定义训练过程
# def train():
#     model.train()
#     for epoch in range(3):  # 训练 3 个 epoch
#         for step, batch in enumerate(train_dataloader):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss

#             optimizer.zero_grad()
#             booster.backward(loss, optimizer)
#             optimizer.step()
#             # scheduler.step()

#             if step % 10 == 0:
#                 logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

# Define a train function

def train_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, _criterion: Callable, 
                train_dataloader: DataLoader, booster: Booster):

    is_pp_last_stage = booster.plugin.stage_manager.is_last_stage()
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    # convert train_dataloader to a iterator
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(range(total_step - 1) ,        # 忽略最后一个批次 做简单处理
              desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]',
              disable=not (is_pp_last_stage)) as pbar:
        # Forward pass
        for _ in pbar:
            outputs = booster.execute_pipeline(train_dataloader_iter,
                                                model,
                                                _criterion,
                                                optimizer,
                                                return_loss=True)
            # Backward and optimize
            if is_pp_last_stage:
                loss = outputs['loss']
                pbar.set_postfix({'loss': loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            # lr_scheduler.step()


# # 开始训练
# if __name__ == "__main__":
#     train()

# Train model
NUM_EPOCHS = 1
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, train_dataloader, booster)