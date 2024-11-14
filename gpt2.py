# --------------------------------------------------------------------------------------------
# ---------------------------------- GPT-2 Model ---------------------------------------------

import math, time, inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_RESIDUAL_SCALE_INIT =  1  # GPT-2 residual scale flag
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Non Trainables : Bias in OpenAI naming convention, but it is actually like a mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()

        # Query, Key, Value projections for all heads
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # # Attention --> Large (T, T) matrix for all the queries and keys
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)

        # y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        
        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_RESIDUAL_SCALE_INIT = 1  # GPT-2 residual scale flag

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Communication
        x = x + self.mlp(
            self.ln_2(x)
        )  # Map - Think individually over what is communicated
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Sharing Scheme
        self.transformer.wte.weight = self.lm_head.weight            ### TODO ###   ### UNDERSTAND THIS ###
        
        # Initialize Weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            std = 0.02
            if hasattr(module, "GPT_RESIDUAL_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        # LayerNorm sclae is already set to 1 and offset to 0
        
        
    def forward(self, idx, targets=None):
        # idx --> (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted"
        
        # Token and Position Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        
        x = tok_emb + pos_emb
        
        # Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Final Layer Norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # (B*T, vocab_size)
        
        return logits, loss

    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"number of decay parameters: {len(decay_params)} , with {num_decay_params} parameters")
        print(f"number of non-decay parameters: {len(nodecay_params)} , with {num_nodecay_params} parameters")

        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused adamw: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


# ------------------------------------------------------------------------------------------
# ------------------------------- GPT-2 Tokenized Data Loader ------------------------------
import os
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        assert split in {"train", "valid"}

        data_root = "/kaggle/input/1b-tokenized-fineweb-edu-text-with-gpt-tokenizer/edu-fineweb-GPT-tokenized-1B/"
        shards = os.listdir(data_root)

        # Filter shards to include only .npy files with the correct split
        shards = [s for s in shards if split in s and s.endswith(".npy")]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards) > 0, "No data found"

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        # Check if we have enough tokens
        if len(buf) < B * T + 1:
            # Move to the next shard if insufficient tokens
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        # Perform reshaping
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes

        return x, y


# ------------------------------------------------------------------------------------------
# --------------------------------- Training GPT-2 Model -----------------------------------

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "Distributed training requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"

    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # Non DDP Run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # Auto detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Using device:", device)


total_batch_size = 524288
B = 8
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "Batch size not divisible by B * T * ddp_world_size"
grad_acc_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print("Total Batch Size:", total_batch_size)
    print("No of Gradient Accumulation Steps:", grad_acc_steps)


# Data Loader
train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

# Set Torch to lower precision (TF32)
# torch.set_float32_matmul_precision('high')

# Create Mode
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model)

# Multi-GPU
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 1900

def get_lr(it):
    # Linear warmup for the first warmup_steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    # For it > lr_decay_steps, return min_lr
    if it > max_steps:
        return min_lr

    # Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (max_lr - min_lr)


# Optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    # Current time
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0.0
    for micro_step in range(grad_acc_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
        loss = loss / grad_acc_steps
        loss_accum += loss.detach()

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_acc_steps - 1)

        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Update weights
    optimizer.step()

    # Let GPU finish
    torch.cuda.synchronize()

    # Current time
    t1 = time.time()
    dt = (t1 - t0)  # in seconds

    # Tokens per second throughput
    token_processed = train_loader.B * train_loader.T * grad_acc_steps * ddp_world_size
    tokens_per_sec = token_processed / dt

    if master_process:
        if step % 25 == 0 or step == max_steps - 1:
            print(f"Step {step}| Loss: {loss_accum.item():.6f} | Norm: {norm:.3f} | LR: {lr:.3e} | Throughput: {tokens_per_sec:.2f} tokens/sec")

        with open("training.log", "a") as f:
            f.write(f"{step},{loss_accum.item()},{norm},{lr},{tokens_per_sec}\n")


# ------------------------------------------------------------------------------------------
# -------------------------- Save and Load GPT-2 Model Weights -----------------------------
# Model name : GPT2-124M-1B-token

# Wait for all processes to finish
if ddp:
    dist.barrier()

# Save the model
if master_process:
    torch.save(model.state_dict(), "GPT2-124M-1B-token.pth")
if ddp:
    destroy_process_group()


# ------------------------------------------------------------------------------------------
# ---------------------- Load and Generate Text from GPT-2 Model ---------------------------
# Model name : GPT2-124M-1B-token
import tiktoken


if master_process:
    # Load the model
    pretrained_model = GPT(GPTConfig(vocab_size=50304))
    pretrained_model.load_state_dict(torch.load("GPT2-124M-1B-token.pth"), strict=False)
    pretrained_model.to(device)


    # Generate Text
    num_return_sequences = 5
    max_length = 32

    enc = tiktoken.get_encoding("gpt2")

    tokens = enc.encode("I was thinking about the meaning of life and")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    x_gen = tokens.to(device)

    while x_gen.size(1) < max_length:
        logits, _ = pretrained_model(x_gen)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, -1, ix)
        x_gen = torch.cat((x_gen, xcol), dim=1)

    print("\n"*5)
    print("="*50)

    for i in range(num_return_sequences):
        tokens = x_gen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"Generated Text {i+1}: {decoded} \n")

    print("-"*50)
