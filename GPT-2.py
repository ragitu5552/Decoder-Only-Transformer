import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    head_size: int = 12
    emb_size: int = 768
    dropout: float = 0.0
    bias: bool = True

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln1 = nn.Linear(config.emb_size, 4*config.emb_size)
        self.glu = nn.GELU(approximate='tanh')
        self.ln2 = nn.Linear(4*config.emb_size, config.emb_size)
    def forward(self, x):
        x = self.ln1(x)
        x = self.glu(x)
        x = self.ln2(x)
        return x

# class head(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.query = nn.Linear(config.emb_size, config.emb_size//config.head_size, bias=False)
#         self.key = nn.Linear(config.emb_size, config.emb_size//config.head_size, bias=False)
#         self.value = nn.Linear(config.emb_size, config.emb_size//config.head_size, bias=False)
#         self.triangle = torch.tril(torch.ones(config.block_size, config.block_size))

#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#         wei = q @ k.transpose(-2, -1)*k.shape[-1]**-0.5
#         wei = wei.masked_fill(self.triangle == 0, float('-inf')) 
#         wei = F.softmax(wei, dim=-1) # (B, T, T)n_embd

#         out = wei @ v
#         return out        

# class selfattention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.heads = nn.ModuleList([head(config) for _ in range(config.head_size)])
#         self.proj = nn.Linear(config.emb_size, config.emb_size)

#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.proj(out)
#         return out

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.emb_size % config.head_size == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.emb_size, 3 * config.emb_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.emb_size, config.emb_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.head_size = config.head_size
        self.emb_size = config.emb_size
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (emb_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.emb_size, dim=2)
        k = k.view(B, T, self.head_size, C // self.head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.head_size, C // self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.head_size, C // self.head_size).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.attention =CausalSelfAttention(config)
        self.ln_f = MLP(config)
        self.ln1 = nn.LayerNorm(config.emb_size)
        self.ln2 = nn.LayerNorm(config.emb_size)
    def forward(self, x):
        x = x + attention(self.ln1(x))
        x = x + MLP(self.ln2(x))
        return x
        
 
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.emb_size),
            wpe = nn.Embedding(config.block_size, config.emb_size),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.emb_size)
        ))
        self.ln_head = nn.Linear(config.emb_size, config.vocab_size, bias=False)
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
    
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, head_size=12, emb_size=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, head_size=16, emb_size=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, head_size=20, emb_size=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, head_size=25, emb_size=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
    
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
    
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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
model = GPT.from_pretrained('gpt2')
print('Weights acquired')