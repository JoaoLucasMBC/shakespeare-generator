import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Multi-head attention with residual & layernorm
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual & layernorm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_len, embedding_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        x = self.embedding(x) + self.position_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits