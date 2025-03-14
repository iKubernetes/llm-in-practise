import torch
from model import MiniGPT, Config, TextDataset
import json

class GPTTester:
    def __init__(self, model_path="minigpt_model.pth"):
        # 加载完整模型信息
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 动态重建配置对象
        self.config = Config()
        for k, v in checkpoint['config'].items():
            setattr(self.config, k, v)
        
        # 初始化模型架构
        self.model = MiniGPT(self.config)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()  # 进入评估模式
        
        # 加载字符映射表
        self.stoi = checkpoint['stoi']
        self.itos = checkpoint['itos']
        
        print(f"成功加载训练于 {len(self.stoi)} 字符的模型")

    def preprocess(self, text):
        """将输入文本转换为模型可处理的张量"""
        # 自动填充/截断序列
        seq = [self.stoi.get(ch, 0) for ch in text][-self.config.seq_len:]
        if len(seq) < self.config.seq_len:
            seq = [0]*(self.config.seq_len - len(seq)) + seq
        
        return torch.tensor(seq).unsqueeze(0)  # 添加batch维度

    def generate(self, prompt, max_len=100, temperature=0.8):
        """文本生成主函数"""
        with torch.no_grad():  # 禁用梯度计算
            generated = []
            input_seq = self.preprocess(prompt)
            
            for _ in range(max_len):
                # 获取模型预测
                logits = self.model(input_seq)
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                
                # 采样下一个字符
                next_char = torch.multinomial(probs, num_samples=1)
                generated.append(next_char.item())
                
                # 更新输入序列
                input_seq = torch.cat([
                    input_seq[:, 1:], 
                    next_char
                ], dim=1)
                
            # 转换生成结果
            return prompt + ''.join([self.itos[idx] for idx in generated])

    def test_output_shape(self):
        """验证模型输出维度"""
        test_input = torch.randint(0, self.config.vocab_size, (1, self.config.seq_len))
        output = self.model(test_input)
        assert output.shape == (1, self.config.seq_len, self.config.vocab_size), \
            f"输出维度错误: 期望 {(1, self.config.seq_len, self.config.vocab_size)}, 实际 {output.shape}"
        print("输出维度验证通过")

if __name__ == "__main__":
    tester = GPTTester()
    
    # 执行维度验证测试
    tester.test_output_shape()
    
    # 示例生成测试
    sample_input = "马哥教育"
    print(f"输入: {sample_input}")
    print(f"生成结果: {tester.generate(sample_input, max_len=50)}")
