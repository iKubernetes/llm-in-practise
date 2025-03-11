import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(16, embed_dim)  # 适配seq_len=16
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        self.register_buffer("dummy_memory", torch.zeros(1, 1, embed_dim))
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        x = self.token_embed(x) + self.pos_embed(pos)
        
        batch_size, seq_len = x.size(0), x.size(1)
        memory = self.dummy_memory.expand(batch_size, seq_len, -1)
        
        for layer in self.layers:
            x = layer(x, memory)
            
        return self.fc(x)
