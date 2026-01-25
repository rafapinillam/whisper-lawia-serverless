# Dockerfile compatible con B200 (Blackwell) y H200 (Hopper)
# NVIDIA NGC PyTorch 25.01+ incluye soporte para sm_100 (Blackwell)
FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

# Instalar ffmpeg (requerido por whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias
# Nota: NO reinstalar torch/torchvision - usar los de NVIDIA NGC
RUN pip install --no-cache-dir \
    openai-whisper>=20231117 \
    requests>=2.31.0 \
    runpod>=1.6.0

# Pre-descargar el modelo Whisper large-v3 durante el build
RUN python -c "import whisper; whisper.load_model('large-v3')"

COPY handler.py .

# Verificar GPU al inicio
CMD ["python", "-u", "handler.py"]
