# 使用 2026 年最新的 vLLM 官方鏡像，內建編譯好的環境
FROM vllm/vllm-openai:latest

# 設定工作目錄
WORKDIR /app

# 安裝 RunPod SDK
RUN pip install --no-cache-dir runpod

# 複製處理程式
COPY handler.py .

# 設定環境變數預設值 (可以在 RunPod 介面覆蓋)
ENV MODEL_PATH="/runpod-volume/model"
ENV TENSOR_PARALLEL=1

# 啟動指令：使用 python 執行 handler.py
# -u 確保日誌即時輸出，方便在 RunPod Log 查看
CMD ["python", "-u", "handler.py"]
