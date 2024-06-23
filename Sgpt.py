import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass
import math
import time
import inspect
import os
import sys

# sys.exit(0)

#hyperparameters
@dataclass
class gptconfig:
    block_size: int = 1024 # maximum context length for predicitons
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 768


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1357)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1357)

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
        self.outproj.NANOGPT_SCLAE_INIT = 1
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

        # att_score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att_score = att_score.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # wei = F.softmax(att_score, dim=-1)
        # out = wei @ v # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention which does the operation of 4 code lines above at once

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.outproj(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.outproj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.outproj.NANOGPT_SCLAE_INIT = 1

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x= self.outproj(x)
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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 # 0.02 come from Xavier init
            if hasattr(module, 'NANOGPT_SCLAE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # load tokens from disk and store them in memory, not in gpu memory
        with open('witcher_elves.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        with open('witcher_tower.txt', 'r', encoding='utf-8') as f:
            text2 = f.read()
        with open('witcher_lady.txt', 'r', encoding='utf-8') as f:
            text3 = f.read()

        text = text + "\n" + text2 + "\n" + text3
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


### old way to load data
# with open('blade-runner-2049.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# text = text[:1000]
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode(text)
# print(len(tokens))
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# but = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

### temp example
# num_setence = 3
# max_length = 40
enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, World!. I am")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_setence, 1)
# x = tokens.to(device)


total_batch_size = 32768 # I'd like to train my model with this batch size ideally, in number of tokens
B = 8 # Due to my low-spec hardware, This is the maximum size
T = 512 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f'total desired batch size: {total_batch_size}')
print(f'=> Calculated gradient accumulation steps: {grad_accum_steps}')


## data loading
train_loader = DataLoaderLite(B=B, T=T)

# torch.set_float32_matmul_precision('high') # precision TF32

## initiate model and get logits
model = BladeGPT(gptconfig())
model.to(device)
# model = torch.compile(model) # triton may not support windows

max_lr = 3e-3
min_lr = max_lr * 0.1
warmup_step = 10
max_steps = 19 # total number of tokens // total_batch_size ~ 1 epoch
def get_lr(it):
    # 1.Linear warmup for warmup_iters step
    if it < warmup_step:
        return max_lr * (it+1) / warmup_step
    # 2.if it > lr_decay_iter, return min lr
    if it > max_steps:
        return min_lr
    # 3.In between, use cosine decay down to min lr
    decay_ratio = (it - warmup_step) / (max_steps - warmup_step)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


## optimize
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device_type=device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))


# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = (t1 - t0) # time difference in second
    token_processed = train_loader.B * train_loader.T * grad_accum_steps
    tok_per_sec = token_processed / dt
    print(f"step {step}, loss: {loss_accum.item():.5f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt:.2f}s, tok/sec: {tok_per_sec:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")

    if step > 0 and (step % 50 == 0 or last_step):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': model.state_dict(),
            'config': model.config,
            'step': step,
            'loss': loss_accum.item()
        }
        # you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)


## Generate tokens
num_sentence = 4
max_length = 32

tokens = enc.encode("What is it like to play with your dog?")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_sentence, 1)
xgen = tokens.to(device)

while xgen.size(1) < max_length:
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)

        topk_prob, topk_idx = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_prob, 1)
        xcol = torch.gather(topk_idx, -1, ix)
        xgen = torch.cat((xgen, xcol), dim=1)

for i in range(num_sentence):
    tokenss = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokenss)
    print(">", decoded)

