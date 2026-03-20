# 베이스 이미지
ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}

WORKDIR /app

# onnxruntime에서 필요한 경우가 있어 설치한다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN pip install --no-cache-dir uv

# 의존성 파일만 먼저 복사 (캐시 최적화)
COPY pyproject.toml ./
COPY uv.lock ./

# uv로 동기화 (lock 기반)
RUN uv sync --frozen --no-dev

# 앱 코드 + 모델 복사
COPY . .

EXPOSE 8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

CMD ["uv", "run", "streamlit", "run", "app.py"]
