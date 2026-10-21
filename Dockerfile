FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX512=OFF" \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-binary llama-cpp-python "llama-cpp-python==0.3.30"
RUN pip install -r requirements.txt

COPY app ./app
COPY bench ./bench
COPY pyproject.toml ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
