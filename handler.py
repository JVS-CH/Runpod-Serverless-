import runpod
from vllm import LLM, SamplingParams
import os

# 1. 取得環境變數或設定預設路徑
# 建議將模型放在 /runpod-volume/qwen3.5-35b-awq 這樣的地方
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/model")
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", 1)) # 如果選 2 張顯卡就設 2

# 2. 初始化 vLLM 引擎
# 針對 122B 模型，gpu_memory_utilization 建議設高，並限制 max_model_len 避免 OOM
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TENSOR_PARALLEL,
    trust_remote_code=True,
    gpu_memory_utilization=0.95, 
    max_model_len=4096, # 視顯存狀況調大 (如 32768)
    enforce_eager=True  # 大模型建議開啟，減少 CUDA graph 佔用的顯存
)

def handler(event):
    '''
    輸入範例: {"input": {"prompt": "你好", "max_tokens": 512}}
    '''
    job_input = event.get("input", {})
    prompt = job_input.get("prompt")
    
    if not prompt:
        return {"error": "No prompt provided"}

    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.7),
        top_p=0.8,
        max_tokens=job_input.get("max_tokens", 1024),
        stop=["<|im_end|>", "<|endoftext|>"] # Qwen 3.5 常用停止詞
    )

    # 執行推論
    outputs = llm.generate([prompt], sampling_params)
    
    # 格式化輸出
    generated_text = outputs[0].outputs[0].text
    
    return {
        "result": generated_text,
        "tokens_generated": len(outputs[0].outputs[0].token_ids)
    }

# 啟動 Serverless 服務
runpod.serverless.start({"handler": handler})
