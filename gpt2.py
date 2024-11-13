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
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # (B*T, vocab_size)
        
        return logits, loss
    
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a HuggingFace/Transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
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
import tiktoken, os
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
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
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B*T

        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard += 1
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B*T

        return x, y


# ------------------------------------------------------------------------------------------
# --------------------------------- Training GPT-2 Model -----------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Using device:", device)


total_batch_size = 131072
B = 2
T = 1024
assert total_batch_size % (B*T) == 0, "Batch size not divisible by B*T"
grad_acc_steps = total_batch_size // (B*T)
print("Total Batch Size:", total_batch_size)
print("No of Gradient Accumulation Steps:", grad_acc_steps)


# Data Loader
train_loader = DataLoaderLite(B, T, split="train")

# Set Torch to lower precision (TF32)
torch.set_float32_matmul_precision('high')

# Get logits and loss
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)


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
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

for step in range(10):
    # Current time
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0.0
    for micro_step in range(grad_acc_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_acc_steps
        loss_accum += loss.detach()
        loss.backward()

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
    token_processed = train_loader.B * train_loader.T * grad_acc_steps
    tokens_per_sec = token_processed / dt
    
    print(f"Step {step}| Loss: {loss_accum.item():.6f} | Norm: {norm:.3f} | LR: {lr:.3e} | Throughput: {tokens_per_sec:.2f} tokens/sec")


# ------------------------------------------------------------------------------------------
# -------------------------- Save and Load GPT-2 Model Weights -----------------------------
# Model name : GPT2-124M-1B-token

# Save the model
torch.save(model.state_dict(), "GPT2-124M-1B-token.pth")
