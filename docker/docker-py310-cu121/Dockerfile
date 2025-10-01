FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=16

# Install base packages like wget, git, etc.
RUN apt update && apt install -y wget git build-essential libxml2 ninja-build \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Install python packages including torch and flash_attn
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install packaging \
    && pip install flash-attn==2.3.6 --no-build-isolation

# Install dependence of Ming
COPY matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl .
RUN pip install --no-cache-dir funasr==1.1.14 flash-attn==2.3.6 peft==0.13.2 diffusers tokenizers==0.20.3 transformers==4.45.0 decord==0.6.0 hyperpyyaml modelscope onnxruntime inflect conformer diffusers lightning gdown openai-whisper==20240930 \
    && pip install --no-cache-dir matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl \
    && pip install --no-cache-dir diffusers==0.33.0 \
    && pip install --no-cache-dir nvidia-cublas-cu12==12.4.5.8 \
    && rm matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl

COPY entrypoint.sh .
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
