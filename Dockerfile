FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Instalar ffmpeg (requerido por whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo Whisper large-v3 durante el build
# Esto evita descargarlo cada vez que el worker inicia (ahorra ~12 segundos)
RUN python -c "import whisper; whisper.load_model('large-v3')"

COPY handler.py .

# Verificar GPU al inicio
CMD ["python", "-u", "handler.py"]
