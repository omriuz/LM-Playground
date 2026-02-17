import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chz
import random
import numpy as np
from tqdm import tqdm


@chz.chz
class TransformerConfig:
    dataset: str = "shakespeare.txt"
    hidden_dim: int = 256
    n_layers: int = 12
    batch_size: int = 64
    n_heads: int = 8

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class PositionalEmbeddings(nn.Module):
    def __init__(self, hidden_dim, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq = float(10000)
        self.seq_len = 1000
        self.device = device
        self.pe = self._get_embs(self.freq,self.seq_len,self.hidden_dim)
        

    def forward(self):
        return self.pe
    
    def _get_embs(self,freq, seq_len, hidden_dim):
        position = torch.arange(seq_len,device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2,device=self.device) *
            (-torch.log(torch.tensor(freq, device=self.device)) / hidden_dim)
        )
        pe = torch.zeros(seq_len, hidden_dim,device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class LayerNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, x, eps=1e-8):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=False)
        x = self.gamma * (x-mean)/torch.sqrt((var+eps)) + self.beta
        return x

class MLPLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln1 = nn.Linear(hidden_dim, 4*hidden_dim)
        self.act_fn = nn.ReLU()
        self.ln2 = nn.Linear(4*hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = self.act_fn(self.ln1(x))
        return self.ln2(x)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, device):
        super().__init__()
        self.Q_w = nn.Linear(hidden_dim,hidden_dim)
        self.K_w = nn.Linear(hidden_dim,hidden_dim)
        self.V_w = nn.Linear(hidden_dim,hidden_dim)
        self.out_w = nn.Linear(hidden_dim,hidden_dim)
        self.device = device
        self.n_heads = n_heads
        
    def forward(self, x):
        B,S,H = x.shape
        assert H % self.n_heads == 0
        head_dim = H // self.n_heads

        q = self.Q_w(x).view(B,S,self.n_heads,head_dim).transpose(-3,-2)
        k = self.K_w(x).view(B,S,self.n_heads,head_dim).transpose(-3,-2)
        v = self.V_w(x).view(B,S,self.n_heads,head_dim).transpose(-3,-2)

        scale = head_dim ** 0.5
        scores = torch.matmul(q,torch.transpose(k,-2,-1))/scale
        mask = torch.triu(torch.ones(S,S,device=self.device),diagonal=1)
        mask = mask.masked_fill(mask.bool(),float("-inf"))

        probs = torch.softmax(scores+mask,dim=-1)
        values = torch.matmul(probs,v).transpose(-2,-3).reshape(B,S,H)
        return self.out_w(values)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, layer_idx, n_heads, device):
        super().__init__()
        self.norm1 = LayerNorm(hidden_dim)
        self.attention = AttentionLayer(hidden_dim, n_heads, device)
        self.norm2 = LayerNorm(hidden_dim)
        self.mlp = MLPLayer(hidden_dim)
        self.device = device
        self.layer_idx = layer_idx
        
    def forward(self, x):
        hid = self.norm1(x)
        hid = self.attention(hid)
        x = x + hid
        hid = self.norm2(x)
        hid = self.mlp(hid)
        return x + hid

class TransformerLLM(nn.Module):
    def __init__(self, vocab_dim, hidden_dim, n_layers,n_heads, device):
        super().__init__()
        self.device=device
        self.embedding = nn.Embedding(vocab_dim,hidden_dim)
        self.pos_embs = PositionalEmbeddings(hidden_dim, device=self.device)
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim, i,n_heads, device) for i in range(n_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_dim, bias=False)
        # self.lm_head.weight = self.embedding.weight

    def forward(self,x):
        hidden = self.embedding(x)
        B, S, H = hidden.shape
        pe = self.pos_embs()[:S]
        hidden = hidden + pe
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        return logits

class Tokenizer():
    def __init__(self, alpahbet):
        self.BOS = "<BOS>"
        self.EOS = "<EOS>"
        self.PAD = "<PAD>"
        self.special_tokens = ["<BOS>","<EOS>","<PAD>"]
        self.alpahbet = self.special_tokens + alpahbet
        self.vocab_size = len(self.alpahbet)

    def _token_to_id(self,token):
        return self.alpahbet.index(token)
    
    def _id_to_token(self, id):
        return self.alpahbet[id]

    def _pretokenize(self, x:str):
        return list(x)
    
    def tokenize(self, x: str, add_eos: bool = True):
        pre = [self.BOS] + list(x)
        if add_eos:
            pre = pre + [self.EOS]
        return [self._token_to_id(t) for t in pre]
    
    def detokenize(self, x:list[int]):
        return [self._id_to_token(id) for id in x]
    
    def is_special_token(self,tok):
        return tok in self.special_tokens
    
    def get_pad_id(self):
        return self._token_to_id(self.PAD)
        

def main(c: TransformerConfig):
    torch.autograd.set_detect_anomaly(True)
    set_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    torch.set_default_device("mps")
    torch.backends.mps.fallback_to_cpu = False

    with open(c.dataset, 'r') as file:
        dataset = file.read()

    alpahbet = list(set(list(dataset)))
    tokenizer = Tokenizer(alpahbet)
    vocab_dim = tokenizer.vocab_size

    model = TransformerLLM(vocab_dim, c.hidden_dim, c.n_layers,c.n_heads, device).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable}")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.get_pad_id())

    ids = torch.tensor(tokenizer.tokenize(dataset), dtype=torch.long, device=device)

    block_size = 512 
    def get_batch(batch_size):
        ix = torch.randint(0, ids.numel() - block_size - 1, (batch_size,), device=device)
        x = torch.stack([ids[i:i+block_size] for i in ix])
        y = torch.stack([ids[i+1:i+block_size+1] for i in ix])
        return x, y
    
    num_steps = 50
    for step in tqdm(range(num_steps)):
        x, y = get_batch(c.batch_size)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_function(logits.reshape(-1,vocab_dim),y.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(loss.detach().item())

    # inference 
    tokens_to_generate = 50
    prompts = ["Was sleeping by","The little Love", "Hey","This brand she"]
    temp = 1
    for prompt in prompts:
        for _ in range(tokens_to_generate):
            tokens = tokenizer.tokenize(prompt, add_eos=False)
            logits = model(torch.tensor([tokens],device=device))
            probs = F.softmax(logits[0, -1, :] / temp, dim=-1)
            new_token_id = torch.multinomial(probs, num_samples=1).item()
            new_token = tokenizer.detokenize([new_token_id])[0]
            prompt = prompt+new_token
            if tokenizer.is_special_token(new_token):
                break
        print(prompt)


if __name__=="__main__":
    chz.nested_entrypoint(main)