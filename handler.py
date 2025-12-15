import runpod
import tempfile
import os
import requests
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("🔄 Cargando Whisper large-v3...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
logger.info("✅ Modelo cargado")

def handler(event):
    try:
        input_data = event.get("input", {})
        audio_url = input_data.get("audio_url")
        language = input_data.get("language", "es")
        task = input_data.get("task", "transcribe")
        
        if not audio_url:
            return {"error": "audio_url requerido"}
        
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
        os.unlink(temp_path)
        
        logger.info(f"✅ {len(text)} caracteres")
        return {
            "text": text,
            "transcription": text,
            "language": getattr(info, 'language', language),
            "duration": getattr(info, 'duration', 0),
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"❌ {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
