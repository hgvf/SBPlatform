FROM python:3.11-slim

# 基本系統工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git && rm -rf /var/lib/apt/lists/*

# 安裝 uv（官方安裝腳本）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /backtest
COPY pyproject.toml /backtest/

# 預同步依賴（建立虛擬環境 .venv）
RUN uv sync --frozen || true

# 其餘專案檔案（為了快取，放在後面）
COPY . /backtest

# 啟動容器
CMD ["bash"]
