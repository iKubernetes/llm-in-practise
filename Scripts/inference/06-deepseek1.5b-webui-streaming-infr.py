import os
import torch
import gradio as gr
from modelscope import AutoTokenizer, AutoModelForCausalLM
from modelscope.hub.snapshot_download import snapshot_download

# For streaming
from transformers import TextIteratorStreamer
import threading
import time # 可以用于调试或模拟生成延迟

# ========== Step 1: 模型准备 ==========
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"模型将被下载或从缓存加载: {'MODELSCOPE_CACHE环境变量或默认ModelScope缓存路径'}")
model_dir = snapshot_download(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("模型加载完成并进入评估模式。")

# ========== Step 2: Prompt构造函数 ==========
def build_prompt(history):
    prompt = ""
    for user, assistant in history:
        prompt += f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n"
    return prompt

# ========== Step 3: 推理函数 (支持流式输出) ==========
def chat(user_input, chat_history, temperature, top_p):
    if chat_history is None:
        chat_history = []

    # 1. 将当前用户输入添加到历史中 (模型回复部分先留空)
    chat_history.append([user_input, ""])

    # 2. 构建完整的 Prompt
    # build_prompt(chat_history[:-1]) 只包含之前的历史，不包含当前用户的回复占位
    # 然后再拼接当前用户的输入
    prompt_history_part = build_prompt(chat_history[:-1])
    full_prompt = f"{prompt_history_part}<|user|>\n{user_input}\n<|assistant|>\n"

    print(f"DEBUG: Full prompt sent to model:\n{full_prompt}")

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # 3. 初始化 TextIteratorStreamer
    # skip_prompt=True 确保流中只包含新生成的token，不重复输入prompt
    # skip_special_tokens=True 避免解码时包含特殊token
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 4. 在单独的线程中运行 model.generate
    # 这样 Gradio UI 不会因为等待生成而冻结
    generation_kwargs = dict(
        inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, # 确保 pad_token_id 与 eos_token_id 一致
        streamer=streamer, # 将 streamer 传递给 generate 方法
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 5. 迭代 streamer，逐步更新UI
    response_so_far = ""
    for new_token in streamer:
        response_so_far += new_token
        # === 清理输出 (应用于当前所有已生成的文本) ===
        # 移除回复中可能意外出现的 <|user|> 或 <|assistant|> 等特殊标记
        cleaned_response_for_display = response_so_far.replace("<|user|>", "").replace("<|assistant|>", "").strip()

        # 更新历史列表的最后一个元素（机器人回复部分）
        # Gradio 的 Chatbot 会根据这个更新进行流式显示
        chat_history[-1][1] = cleaned_response_for_display
        
        # 实时返回当前的聊天历史和状态，Gradio 会更新界面
        yield chat_history, chat_history

    # 当流式生成结束后，chat_history[-1][1] 已经包含最终的完整回复
    # Gradio 会在生成完成后自动停止更新，无需额外的return语句。

# ========== Step 4: 构建 Gradio 界面 ==========
with gr.Blocks(title="DeepSeek-Qwen 多轮对话 (流式输出)") as demo:
    gr.Markdown("## DeepSeek-Qwen 多轮对话界面\n支持 ModelScope 加载、上下文记忆、Gradio 聊天 UI，**流式输出**。")

    # 聊天机器人显示区域
    chatbot = gr.Chatbot(show_label=False, avatar_images=(None, "robot_avatar.png")) # 可以添加机器人头像

    # 用户输入框
    user_input = gr.Textbox(placeholder="请输入问题，按回车发送", show_label=False, container=False)

    # 存储会话历史的状态变量
    state = gr.State([])

    with gr.Row():
        # 生成参数滑块
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7, label="Temperature", scale=1)
        top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.95, label="Top P", scale=1)
        
        # 清除按钮
        clear_button = gr.ClearButton(
            components=[user_input, chatbot], # 点击时清空输入框和聊天记录显示
            value="清除会话", # 按钮文本
            size="lg" # 按钮大小
        )
        # 清除按钮的额外操作：清空state（会话历史）
        # 当clear_button被点击时，触发一个lambda函数，将state重置为空列表
        clear_button.click(lambda: [], outputs=[state])


    # 绑定用户输入提交事件
    user_input.submit(
        chat,
        inputs=[user_input, state, temperature_slider, top_p_slider],
        outputs=[chatbot, state]
    )
    # 提交后清空输入框
    user_input.submit(lambda: "", None, user_input)

# ========== Step 5: 启动服务 ==========
if __name__ == "__main__":
    print("\nGradio 服务正在启动...")
    print("请访问控制台输出的URL以使用Web UI。")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)