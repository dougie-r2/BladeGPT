import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass
import math

#hyperparameters
@dataclass
class gptconfig:
    block_size: int = 64 #maximum context length for predicitons
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 3
    n_head: int = 4
    n_embed: int = 128

torch.manual_seed(1357)

# class Head(nn.Module):
#     "one head of sef-attention"

#     def __init__(self, config):
#         super().__init__()
#         self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
#         self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
#         self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

#     def forward(self, x):
#         # input of size (batch, time-step, channels)
#         # output of size (batch, time-step, head size)
#         B, T, C = x.shape
#         k = self.key(x)
#         q = self.query(x)

#         # compute atention scores (affinities)
#         wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#         wei = F.softmax(wei, dim=-1)
#         wei = self.dropout(wei)
#         # perform the weighted aggregation of the values
#         v = self.value(x)
#         out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
#         return out


# class MultiHead(nn.Module):
#     """ multiple heads of self-attention in parallel """

#     def __init__(self, config):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(config.head_size) for _ in range(config.num_heads)])
#         self.proj = nn.Linear(config.head_size * config.num_heads, config.n_embed)
#         self.dropout = nn.Dropout(config.dropout)

#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.dropout(self.proj(out))
#         return out

class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.outproj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att_score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_score = att_score.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        wei = F.softmax(att_score, dim=-1)
        out = wei @ v # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.outproj(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x= self.proj(x)
        return x

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ma = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.ma(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class BladeGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            blocks =  nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed), # Final Layernorm
        ))
        
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) 

    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T <= self.config.block_size

        tok_emb = self.transformer.wte(idx) # B, T, C
        pos_emb = self.transformer.wpe(torch.arange(0, T, dtype=torch.long, device=idx.device)) # T, C
        x = tok_emb + pos_emb # B, T, C

        for block in self.transformer.blocks:
            x = block(x) # B, T, C
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # B, T, vocab_size
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits
    




device = 'cuda' if torch.cuda.is_available() else 'cpu'
## Read Data
with open('blade-runner-2049.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print(text[:1000])


## tiktoken
num_setence = 3
max_length = 40
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, World!. I am")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_setence, 1)
x = tokens.to(device)
# tokens = enc.encode(text)
print(len(tokens))

# B, T = 4, 4
# buf = torch.tensor(tokens[:B*T + 1])
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

# x = x.to(device)
# y = y.to(device)


## Train and test split

## data loading

## initiate model and get logits
model = BladeGPT(gptconfig())
model.to(device)
# logits, loss = model(x, y)

# print(loss)


## Generate tokens
torch.manual_seed(1357)
torch.cuda.manual_seed(1357)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        topk_prob, topk_idx = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_prob, 1)
        xcol = torch.gather(topk_idx, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_setence):
    tokenss = x[i, :max_length].tolist()
    decoded = enc.decode(tokenss)
    print(">", decoded)
import sys; sys.exit(0)