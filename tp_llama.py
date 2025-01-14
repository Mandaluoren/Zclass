import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Callable
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam
import colossalai
from colossalai.accelerator import get_accelerator
import torch.distributed as dist
# Define config parameters
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
MAX_LENGTH = 128

# Initialize ColossalAI
colossalai.launch_from_torch()

# Define 'criterion' function to calculate loss
def _criterion(outputs, inputs):
    return outputs.loss

# Load Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print("Dataset loaded successfully")

# Load LLaMA tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('./llama7bconfig')  # Ensure this path points to a valid tokenizer config
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
cfg = LlamaConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_hidden_layers=32,
    intermediate_size=11008,
    rms_norm_eps=1e-6,
    use_cache=False,
    vocab_size=32000,
    max_position_embeddings=2048
)
model = LlamaForCausalLM(config=cfg)  # Initialize LLaMA model from scratch
print(model)
print("LLaMA model initialized")

# Preprocessing function for Wikitext
def preprocess_function(examples):
    """
    Tokenize the text data for causal language modeling.
    """
    return tokenizer(
        examples["text"],  # Wikitext dataset has a "text" column
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
print("Dataset tokenized")

# Convert tokenized dataset to PyTorch Dataset
class WikitextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.input_ids = torch.tensor(dataset["input_ids"])
        self.attention_mask = torch.tensor(dataset["attention_mask"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx]  # For causal language modeling, labels are shifted input_ids
        }

# Prepare train dataset and dataloader
train_dataset = WikitextDataset(tokenized_datasets["train"])
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#  assert self.batch_size == self.microbatch_size * self.num_microbatches
print("Dataloader prepared")

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

# Use ColossalAI's HybridParallelPlugin
plugin = HybridParallelPlugin(tp_size=8, pp_size=1, microbatch_size=1, num_microbatches=8)
booster = Booster(plugin=plugin)

# Boost model, optimizer, and dataloader
model, optimizer, _criterion, _, _ = booster.boost(model, optimizer=optimizer, criterion=_criterion)
print(model)
# Get distributed logger
logger = get_dist_logger()

# def move_to_cuda(batch):
#     return {k: v.to(get_accelerator().get_current_device()) for k, v in batch.items()}

# Training function
def train_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, _criterion: Callable, 
                train_dataloader: DataLoader, booster: Booster):
    # is_pp_last_stage = booster.plugin.stage_manager.is_last_stage()
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(range(total_step - 2),  # Ignore the last batch to simplify pipeline handling
              desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]",
              disable=not dist.get_rank()==0
              # disable=not is_pp_last_stage
             ) as pbar:
        for _ in pbar:
            # # Forward pass using ColossalAI's pipeline
            # outputs = booster.execute_pipeline(train_dataloader_iter,
            #                                    model,
            #                                    _criterion,
            #                                    optimizer,
            #                                    return_loss=True)
            # # # Backward and optimize
            # # if is_pp_last_stage:
            # #     loss = outputs["loss"]
            # #     pbar.set_postfix({"loss": loss.item()})
            data = next(train_dataloader_iter)
            data = data.cuda()
            outputs = model(**data)
            loss = _criterion(outputs, None)
            # Backward
            booster.backward(loss, optimizer)
            pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()

# Train the model
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, optimizer, _criterion, train_dataloader, booster)

print("Training complete")
