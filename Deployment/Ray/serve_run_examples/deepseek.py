from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

# 初始化 LLMConfig 对象，用于定义模型的加载、部署和运行时配置
llm_config = LLMConfig(
    # 模型加载配置 (model_loading_config) 
    model_loading_config={
        "model_id": "deepseek",
        "model_source": "/Models/Pretrained_Models/DeepSeek-R1-0528-Qwen3-8B",
    },
    
    # 部署配置 (deployment_config)
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 1,
        }
    },

    # 运行时环境配置：定义部署运行时需要的环境变量、依赖项等
    # 这里是告诉 vLLM 使用其 v1 版本的功能或兼容性模式
    runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
    
    # 引擎参数 (engine_kwargs)： 这些参数直接传递给底层 LLM 推理引擎（通常是 vLLM），用于优化性能
    engine_kwargs={
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.92,
        "dtype": "auto",
        "max_num_seqs": 40,
        "max_model_len": 16384,
        # 启用分块预填充：允许大型输入在 GPU 上分批处理
        "enable_chunked_prefill": True,
        # 启用前缀缓存：加速具有相同前缀的请求（如在对话场景中）
        "enable_prefix_caching": True,
    },
)

# 使用 build_openai_app 函数构建一个完整的 Ray Serve 应用
# 这个应用会暴露一个与 OpenAI API 兼容的 HTTP 接口 (如 /v1/chat/completions)
# 它接受一个包含 LLMConfig 列表的字典
llm_app = build_openai_app({"llm_configs": [llm_config]})

# 如果您希望在脚本内部启动部署，可以使用 serve.run()
# 否则，通常使用 `serve deploy <script>:llm_app` 命令从命令行启动
#serve.run(llm_app, route_prefix="/")
