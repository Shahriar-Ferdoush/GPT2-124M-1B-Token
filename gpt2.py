import math, time
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





# ------------------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open("input.txt", "r") as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        print("Total tokens:", len(self.tokens))
        print("Total batches:", len(self.tokens) // (B*T))
        
        # State
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_position += B*T
        
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
            
        return x, y




# ------------------------------------------------------------------------------------------
# Training
num_return_sequences = 5
max_length = 30

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Using device:", device)


# Data Loader
train_loader = DataLoaderLite(B=2, T=1024)

# Set Torch to lower precision (TF32)
torch.set_float32_matmul_precision('high')

# Get logits and loss
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for i in range(10):
    # Current time
    t0 = time.time()
    
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    
    loss.backward()
    
    optimizer.step()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Let GPU finish
    torch.cuda.synchronize()
    
    # Current time
    t1 = time.time()
    dt = (t1 - t0) * 1000   # in ms
    
    # Tokens per second throughput
    tokens_per_sec = train_loader.B * train_loader.T / (t1 - t0)
    
    print(f"Step {i}, Loss: {loss.item()}, Time: {dt:.2f} ms, Norm: {norm:.4f}, Tokens/sec: {tokens_per_sec:.2f}")
    



































import sys; sys.exit(0)
model.eval()
model.to(device)

# prefix token
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# generate
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    logits = model(x)
    logits = logits[:, -1, :]
    
    probs = F.softmax(logits, dim=-1)
    
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    
    ix = torch.multinomial(topk_probs, num_samples=1)
    xcol = torch.gather(topk_indices, -1, ix)
    x = torch.cat((x, xcol), dim=1)
    

for i in range (num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print("Generated: ", decoded)