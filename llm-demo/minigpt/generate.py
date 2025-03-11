import torch
from model import MiniGPT

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = MiniGPT(
        vocab_size=len(checkpoint["char2idx"]),
        embed_dim=checkpoint["config"]["embed_dim"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["char2idx"]

def generate_text(model, char2idx, start_str, max_len=50):
    idx2char = {v: k for k, v in char2idx.items()}
    device = next(model.parameters()).device
    
    input_seq = [char2idx[ch] for ch in start_str]
    generated = list(start_str)
    
    with torch.no_grad():
        for _ in range(max_len):
            inputs = torch.tensor([input_seq[-16:]]).to(device)
            outputs = model(inputs)
            next_char = idx2char[outputs.argmax(-1)[0, -1].item()]
            generated.append(next_char)
            input_seq.append(char2idx[next_char])
    
    return ''.join(generated)

if __name__ == "__main__":
    model, char2idx = load_model("mg_edu_gpt.pth")
    print(generate_text(model, char2idx, "马哥"))
