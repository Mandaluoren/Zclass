# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# from torch.utils.data import DataLoader
# from transformers import default_data_collator
# from datasets import load_dataset

# # 加载MRPC数据集
# # dataset = load_dataset("glue", "mrpc")


# # 加载数据集
# data_files = {
#     'train': '/root/chj/gees/GeeSibling/examples/datasets/mrpc/train.jsonl',
#     'test': '/root/chj/gees/GeeSibling/examples/datasets/mrpc/test.jsonl',
#     'validation': '/root/chj/gees/GeeSibling/examples/datasets/mrpc/validation.jsonl'
# }

# dataset = load_dataset('json', data_files=data_files)


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # 数据预处理
# def preprocess_function(examples):
#     return tokenizer(
#         examples["sentence1"], 
#         examples["sentence2"], 
#         truncation=True, 
#         padding="max_length", 
#         max_length=128
#     )

# encoded_dataset = dataset.map(preprocess_function, batched=True)
# encoded_dataset = encoded_dataset.remove_columns(["sentence1", "sentence2", "idx"])
# encoded_dataset.set_format("torch")

# # 准备训练和验证数据
# train_loader = DataLoader(encoded_dataset["train"], batch_size=8, shuffle=True, collate_fn=default_data_collator)

# # 加载模型
# # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).cuda()
# model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2).cuda()

# # 定义优化器
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# # 训练循环
# epochs = 1
# model.train()
# from tqdm import tqdm
# for epoch in range(epochs):
#     # 使用 tqdm 包装 DataLoader
#     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
#     for batch in progress_bar:
#         # 将数据移动到 GPU
#         input_ids = batch["input_ids"].cuda()
#         attention_mask = batch["attention_mask"].cuda()
#         labels = batch["labels"].cuda()

#         # 前向传播
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # 更新 tqdm 的显示内容
#         progress_bar.set_postfix({"Loss": loss.item()})

# # for epoch in range(epochs):
# #     for batch in train_loader:
# #         # 将数据移动到GPU
# #         input_ids = batch["input_ids"].cuda()
# #         attention_mask = batch["attention_mask"].cuda()
# #         labels = batch["labels"].cuda()

# #         # 前向传播
# #         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
# #         loss = outputs.loss

# #         # 反向传播和优化
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         print(f"Epoch: {epoch}, Loss: {loss.item()}")


import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import default_data_collator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the Wikitext dataset from a local path
save_path = "./wikitext-local"
dataset = load_from_disk(save_path)
print("Dataset loaded from local:", dataset)

# Inspect the dataset structure
print("Dataset structure:")
print(dataset)

# Load LLaMA 2 tokenizer and model
# model_name = "meta-llama/Llama-2-7b-hf"  # Ensure the correct model path or name
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

# tokenizer = LlamaTokenizer.from_pretrained('/mnt/fs/model/Llama-2-7b-hf')
from transformers import LlamaTokenizer
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained('./llama7bconfig')
cfg = LlamaConfig()
model = LlamaForCausalLM(config=cfg).cuda()
print("获取到model")

# Tokenize the dataset
def preprocess_function(examples):
    """
    Tokenize the text data for causal language modeling.
    """
    return tokenizer(
        examples["text"],  # The Wikitext dataset has a "text" column
        truncation=True,
        padding="max_length",
        max_length=128  # Adjust the max length based on your model and GPU memory
    )

# Apply preprocessing to the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Remove the "text" column to avoid redundancy after tokenization
encoded_dataset = encoded_dataset.remove_columns(["text"])

# Set dataset format to PyTorch tensors
encoded_dataset.set_format("torch")

# Prepare the DataLoader for training
train_loader = DataLoader(
    encoded_dataset["train"],  # Training split
    batch_size=8,              # Adjust batch size based on your GPU
    shuffle=True,
    collate_fn=default_data_collator  # Collate function for batching
)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 1
model.train()

for epoch in range(epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
    for batch in progress_bar:
        # Move data to GPU
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        # Shift input_ids for causal language modeling
        labels = input_ids.clone()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar with the current loss
        progress_bar.set_postfix({"Loss": loss.item()})

print("Training complete.")