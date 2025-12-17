import runpod
import tempfile
import os
import requests
import logging
import torch
import whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🔧 Dispositivo: {device}")
if device == "cuda":
    logger.info(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"🔧 CUDA Version: {torch.version.cuda}")

# Cargar modelo Whisper original de OpenAI (usa PyTorch directamente, no CTranslate2)
logger.info("🔄 Cargando Whisper large-v3 en GPU...")
model = whisper.load_model("large-v3", device=device)
logger.info("✅ Modelo cargado en GPU")

def handler(event):
    try:
        input_data = event.get("input", {})
        audio_url = input_data.get("audio_url")
        language = input_data.get("language", "es")
        task = input_data.get("task", "transcribe")
        
        if not audio_url:
            return {"error": "audio_url es requerido"}
        
        logger.info(f"📥 Descargando: {audio_url}")
        response = requests.get(audio_url, timeout=300, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            temp_path = f.name
        
        logger.info("🎤 Transcribiendo con GPU...")
        result = model.transcribe(
            temp_path,
            language=language,
            task=task,
            fp16=(device == "cuda")  # Usar FP16 en GPU para máxima velocidad
        )
        
        text = result["text"]
        detected_language = result.get("language", language)
        
        # Calcular duración aproximada
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        logger.info(f"✅ Transcripción completada: {len(text)} caracteres, {duration:.1f}s")
        return {
            "text": text,
            "transcription": text,
            "language": detected_language,
            "duration": duration,
            "status": "completed",
            "device": device
        }
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
