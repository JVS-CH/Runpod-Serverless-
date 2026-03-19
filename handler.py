import runpod
from vllm import LLM, SamplingParams

# 初始化模型（這是在 Worker 啟動時執行的，避免重複加載）
# 確保路徑指向你的 Network Volume
MODEL_PATH = "/runpod-volume/Qwen3.5-35B-A3B"

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.92,
    max_model_len=8192
)

def handler(event):
    """
    RunPod 傳入的是一個 job 物件，輸入資料在 job["input"]
    """
    input_data = event["input"]
    prompt = input_data.get("prompt", "你好")
    
    sampling_params = SamplingParams(
        temperature=input_data.get("temperature", 0.7),
        max_tokens=input_data.get("max_tokens", 1024)
    )

    outputs = llm.generate([prompt], sampling_params)
    return {"text": outputs[0].outputs[0].text}

# 這是官方手冊要求的啟動方式
runpod.serverless.start({"handler": handler})
