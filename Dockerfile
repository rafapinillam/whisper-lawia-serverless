 Dockerfile para Faster Whisper Large V3
# Versión: v2
# Fecha: 2026-01-27
# Cambio: Migración de openai-whisper a faster-whisper para 4x velocidad y menor costo
# GPU: Compatible con B200 (Blackwell sm_100), H200 (Hopper), A100, L40S

# NVIDIA NGC PyTorch 25.01+ incluye soporte para Blackwell (sm_100)
FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Instalar ffmpeg (requerido por whisper para procesar audio)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

#Instalar faster-whisper y dependencias
# Nota: NO reinstalar torch - usar el de NVIDIA NGC que tiene soporte Blackwell
RUN pip install --no-cache-dir \
    faster-whisper>=1.0.0 \
    requests>=2.31.0 \
    runpod>=1.6.0

# Pre-descargar el modelo Faster Whisper large-v3 durante el build
# Esto evita descargarlo cada vez que el worker inicia (reduce cold start)
# Usamos CPU para el download, en runtime usará GPU con float16
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

COPY handler.faster-whisper.py handler.py

CMD ["python", "-u", "handler.py"]
