import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

#hyperparameters
batch_size = 16
block_size = 64 #maximum context length for predicitons
learning_rate = 3e-4
vocab_size = 50257 # temp

n_embed = 128
n_head = 4
n_layer = 3
dropout = 0.2

torch.manual_seed(1357)

class Head(nn.Module):
    "one head of sef-attention"

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute atention scores (affinities)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHead(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.ma = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.ma(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class BladeGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embed)
        self.wpe = nn.Embedding(block_size, n_embed)
        self.blocks =  nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.size()

        tok_emb = self.wte(idx) # B, T, C
        pos_emb = self.wpe(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_emb # B, T, C
        x = self.blocks(x) # B, T, C
        x = self.ln_f(x) # B, T, C
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
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)
print(len(tokens))

B, T = 4, 4
buf = torch.tensor(tokens[:B*T + 1])
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

x = x.to(device)
y = y.to(device)

print(x.size())
print(y.size())


## Train and test split

## data loading

## initiate model and get logits
model = BladeGPT()
model.to(device)
logits, loss = model(x, y)

print(loss)
import sys; sys.exit(0)