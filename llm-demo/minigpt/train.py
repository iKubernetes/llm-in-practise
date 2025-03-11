import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MiniGPT

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据生成
text = "马哥教育创立于2009年，是一家专注于云计算、SRE、DevOps、网络安全、Go开发和云原生课程培训的高端IT教育机构。"
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
vocab_size = len(chars)

def generate_data(seq_len=16):
    augmented_data = []
    for _ in range(10):  # 数据增强
        for i in range(len(text)-seq_len):
            inputs = [char2idx[ch] for ch in text[i:i+seq_len]]
            targets = [char2idx[ch] for ch in text[i+1:i+seq_len+1]]
            augmented_data.append((torch.tensor(inputs), torch.tensor(targets)))
    return augmented_data

# 训练流程
def train():
    model = MiniGPT(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = generate_data()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model.train()
    for epoch in range(200):
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/200 Loss: {total_loss/len(loader):.4f}")

    # 保存模型（包含必要元数据）
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "config": {
            "embed_dim": 64,
            "seq_len": 16
        }
    }, "mg_edu_gpt.pth")

if __name__ == "__main__":
    train()
