import runpod
import tempfile
import os
import requests
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Usar float32 que NO requiere cuDNN
# float16 y int8 requieren cuDNN que no está disponible en todas las imágenes
logger.info("🔄 Cargando Whisper large-v3 (float32 - sin cuDNN)...")
try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float32")
    logger.info("✅ Modelo cargado en GPU con float32")
except Exception as e:
    logger.warning(f"⚠️ GPU falló ({e}), usando CPU...")
    model = WhisperModel("large-v3", device="cpu", compute_type="float32")
    logger.info("✅ Modelo cargado en CPU")

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
        
        logger.info("🎤 Transcribiendo...")
        segments, info = model.transcribe(temp_path, language=language, task=task)
        
        text = " ".join([segment.text for segment in segments])
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        logger.info(f"✅ Transcripción completada: {len(text)} caracteres")
        return {
            "text": text,
            "transcription": text,
            "language": getattr(info, 'language', language),
            "duration": getattr(info, 'duration', 0),
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
