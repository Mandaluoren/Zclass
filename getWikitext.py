from datasets import load_dataset
import os
from datasets import load_dataset, load_from_disk
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
# print("Get done")
# print(dataset)

# # 加载数据集
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
# print("Dataset loaded:", dataset)

# # 创建保存目录
save_path = "./wikitext-local"
# os.makedirs(save_path, exist_ok=True)

# # 将数据集保存到本地
# dataset.save_to_disk(save_path)
# print(f"Dataset saved to {save_path}")

# 验证:从本地加载数据集
# loaded_dataset = load_dataset("wikitext-local", data_dir=save_path)
save_path = "./wikitext-local"
loaded_dataset = load_from_disk(save_path)
print("Loaded from local:", type(loaded_dataset))
print(loaded_dataset['train'])