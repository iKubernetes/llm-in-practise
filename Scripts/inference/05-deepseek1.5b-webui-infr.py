import os
import torch
import gradio as gr
from modelscope import AutoTokenizer, AutoModelForCausalLM
from modelscope.hub.snapshot_download import snapshot_download

# ========== Step 1: 模型准备 ==========
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_dir = snapshot_download(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ========== Step 2: Prompt构造函数 ==========
def build_prompt(history):
    prompt = ""
    for user, assistant in history:
        prompt += f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n"
    return prompt

# ========== Step 3: 推理函数 ==========
def chat(user_input, chat_history, temperature, top_p):
    if chat_history is None:
        chat_history = []

    prompt = build_prompt(chat_history)
    full_prompt = f"{prompt}<|user|>\n{user_input}\n<|assistant|>\n"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        output_ids[-1][inputs["input_ids"].shape[-1]:],  # 修正索引以处理batch size为1的情况
        skip_special_tokens=True
    ).strip()

    # ========== Step 3.1: 清理输出 ==========
    # 移除回复中可能意外出现的 <|user|> 或 <|assistant|> 等特殊标记
    response = response.replace("<|user|>", "").replace("<|assistant|>", "").strip()

    chat_history.append((user_input, response))
    return chat_history, chat_history

# ========== Step 4: 构建 Gradio 界面 ==========
with gr.Blocks(title="DeepSeek-Qwen 多轮对话") as demo:
    gr.Markdown("## DeepSeek-Qwen 多轮对话界面\n支持 ModelScope 加载、上下文记忆、Gradio 聊天 UI。")

    chatbot = gr.Chatbot(show_label=False)
    user_input = gr.Textbox(placeholder="请输入问题，按回车发送", show_label=False)

    with gr.Row():
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="Temperature")
        top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.95, label="Top P")

    state = gr.State([])  # chat history

    user_input.submit(
        chat,
        inputs=[user_input, state, temperature_slider, top_p_slider],
        outputs=[chatbot, state]
    )
    user_input.submit(lambda: "", None, user_input)  # 清空输入框

# ========== Step 5: 启动服务 ==========
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)