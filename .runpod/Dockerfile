# 使用 vLLM 官方鏡像
FROM vllm/vllm-openai:latest

WORKDIR /app

# 安裝必要套件
RUN pip install --no-cache-dir runpod

COPY handler.py .

ENV MODEL_PATH="/runpod-volume/model"
ENV TENSOR_PARALLEL=1

# --- 關鍵修正處 ---
# 強制重設 ENTRYPOINT 為空，確保 CMD 能獨立執行
ENTRYPOINT []

# 啟動指令
CMD ["python3", "-u", "handler.py"]