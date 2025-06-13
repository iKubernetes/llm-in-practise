import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Config:
    seq_len = 256        # 输入序列长度
    n_layer = 4         # Transformer层数
    n_head = 4          # 注意力头数
    embed_dim = 128     # 嵌入维度
    dropout = 0.1       # Dropout概率
    lr = 3e-4           # 学习率
    weight_decay = 0.1  # L2正则化系数
    epochs = 200        # 训练轮次
    batch_size = 2      # 批次大小

class TextDataset(Dataset):
    def __init__(self, text, seq_len, config):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        
        # 动态设置词汇量
        config.vocab_size = len(chars)  
        
        data = [self.stoi[ch] for ch in text]
        assert all(0 <= idx < config.vocab_size for idx in data), "存在超出词汇表的字符"
        
        self.x, self.y = [], []
        for i in range(len(data)-seq_len):
            self.x.append(data[i:i+seq_len])
            self.y.append(data[i+1:i+1+seq_len])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.embed_dim))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.n_head,
                dim_feedforward=4*config.embed_dim,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.n_layer
        )
        self.ln = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed[:, :T, :]
        x = tok_emb + pos_emb
        x = self.transformer(x)
        x = self.ln(x)
        return self.head(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    
    # 训练数据
    text = "马哥教育创立于2009年，是一家专注于云计算、SRE、DevOps、网络安全、Go开发和云原生课程培训的高端IT教育机构。"
    
    # 动态初始化数据集
    dataset = TextDataset(text, config.seq_len, config)  
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = MiniGPT(config).to(device)
    
    # 优化器带L2正则化
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # 训练循环
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, config.vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {total_loss/len(loader):.4f}")

    # 保存完整模型信息
    torch.save({
        "model_state": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
        "config": vars(config)
    }, "minigpt_model.pth")
