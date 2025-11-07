from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# 1. 加载模型
model_path = "./Qwen3-4B-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

llm = LLM(
    model=model_path,
    quantization="compressed-tensors",
    dtype="half",
    max_model_len=32768,
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    enforce_eager=True,
    disable_log_stats=True,
    disable_custom_all_reduce=True,
)

# 2. 正确采样参数
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
    logprobs=1,           # 必须是 int
    prompt_logprobs=None, # 必须是 None
)

def compute_eternal_perplexity(llm, tokenizer, dataset):
    perplexities = []
    
    for entry in tqdm(dataset, desc="ETERNAL PPL"):
        instruction = (entry.get("instruction","") + "\n" + entry.get("input","")).strip()
        answer = entry.get("output", "")
        if not answer:
            continue
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 真正生成
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        # 正确提取 logprobs 
        logprobs_float = []
        for step_dict in output.outputs[0].logprobs[1:]:  # 跳过第一个 token
            if not step_dict:  # 空 dict
                continue
            # 取概率最高的 token 的 logprob
            top_token_id = max(step_dict.keys(), key=lambda k: step_dict[k].logprob)
            logprobs_float.append(step_dict[top_token_id].logprob)
        
        if len(logprobs_float) == 0:
            continue
            
        nll = -np.mean(logprobs_float)
        ppl = np.exp(nll)
        perplexities.append(ppl)
    
    return np.mean(perplexities), np.std(perplexities)

# 3. 执行
print("启动 Perplexity 评估...")
dataset = load_dataset("llm-wizard/alpaca-gpt4-data-zh", split="train[128:256]")

mean_ppl, std_ppl = compute_eternal_perplexity(llm, tokenizer, dataset)

print(f"\n评估完成！")
print(f"真实标准 Perplexity: {mean_ppl:.3f} ± {std_ppl:.3f}")
print(f"原模型参考值 ≈ 8.19")
print(f"上升 {mean_ppl - 8.19:.3f} → ", end="")
if mean_ppl < 9.0:
    print("量化完美无损！")
else:
    print("建议重新校准")
