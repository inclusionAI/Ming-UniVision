FROM python:3.8

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=16
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Install base packages like wget, git, etc.
RUN apt update && apt install -y wget git build-essential libxml2 ninja-build \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Install python packages including torch and flash_attn
RUN wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run \
    && sh cuda_12.2.2_535.104.05_linux.run --toolkit --silent --override \
    && echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf \
    && echo '/usr/local/cuda-12.2/lib64' >> /etc/ld.so.conf \
    && ldconfig \
    && rm cuda_12.2.2_535.104.05_linux.run

RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    && pip install packaging \
    && pip install flash-attn==2.3.6 --no-build-isolation
#  --index-url https://download.pytorch.org/whl/cu122 

# Install dependence of Ming
RUN git clone --depth 1 https://github.com/inclusionAI/Ming.git \
    && cd Ming \
    && pip install --no-cache-dir funasr==1.1.14 flash-attn==2.3.6 peft==0.13.2 diffusers tokenizers==0.20.3 transformers==4.45.0 decord==0.6.0 hyperpyyaml modelscope onnxruntime inflect conformer diffusers lightning gdown openai-whisper==20240930 \
    && pip install --no-cache-dir data/matcha_tts-0.0.5.1-cp38-cp38-linux_x86_64.whl \
    && pip install --no-cache-dir diffusers==0.33.0 \
    && pip install --no-cache-dir nvidia-cublas-cu12==12.4.5.8

COPY entrypoint.sh .
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
