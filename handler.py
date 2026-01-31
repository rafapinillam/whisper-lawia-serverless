"""
Handler para Faster Whisper Large V3 en RunPod Serverless
Versión: v2
Fecha: 2026-01-30
Cambio: Preservar timestamps de segmentos para citas reproducibles

Beneficios:
- 4x más rápido que whisper original
- Mismo modelo large-v3 (misma precisión)
- Menor uso de VRAM
- Menor costo por transcripción
- [v2] Retorna segments_json con timestamps para evidence_snippets
"""

import runpod
import tempfile
import os
import requests
import logging
from faster_whisper import WhisperModel
from urllib.parse import urlparse
import ipaddress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dominios permitidos para descargar audio (anti-SSRF)
DEFAULT_ALLOWED_DOMAINS = [
    "supabase.co",
    "supabase.in",
    "supabase.com",
    "files.lawia.app",
]

DEFAULT_ALLOWED_DOMAIN_SUFFIXES = [
    ".supabase.co",
    ".supabase.in",
    ".supabase.com",
    ".backblazeb2.com",
]

def _parse_csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    if not raw:
        return []
    return [x.strip().lower() for x in raw.split(",") if x.strip()]

ALLOWED_DOMAINS = sorted(set(DEFAULT_ALLOWED_DOMAINS + _parse_csv_env("LAWIA_ALLOWED_AUDIO_DOMAINS")))
ALLOWED_DOMAIN_SUFFIXES = sorted(set(DEFAULT_ALLOWED_DOMAIN_SUFFIXES + _parse_csv_env("LAWIA_ALLOWED_AUDIO_SUFFIXES")))


def is_safe_url(url: str) -> tuple[bool, str]:
    """
    Valida que una URL sea segura para descargar.
    Previene SSRF bloqueando IPs privadas y dominios no permitidos.
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False, f"Esquema no permitido: {parsed.scheme}"

        hostname = parsed.hostname
        if not hostname:
            return False, "URL sin hostname"

        # Verificar si es una IP
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"IP no permitida: {hostname}"
            if str(ip).startswith("169.254."):
                return False, f"IP de metadata bloqueada: {hostname}"
        except ValueError:
            pass

        hostname_lower = hostname.lower()

        if hostname_lower in ALLOWED_DOMAINS:
            return True, "Dominio permitido"

        for suffix in ALLOWED_DOMAIN_SUFFIXES:
            if hostname_lower.endswith(suffix):
                return True, f"Subdominio permitido: {suffix}"

        return False, f"Dominio no permitido: {hostname}"

    except Exception as e:
        return False, f"Error validando URL: {str(e)}"


# Detectar GPU y configurar compute_type
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configurar compute_type según GPU disponible
# - float16: Para GPUs modernas (Ampere, Hopper, Blackwell) - máxima velocidad
# - int8: Fallback para GPUs antiguas o CPU
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"🔧 GPU detectada: {gpu_name}")

    # Usar float16 para GPUs modernas (mejor rendimiento)
    compute_type = "float16"
    logger.info(f"🔧 Compute type: {compute_type}")
else:
    compute_type = "int8"
    logger.info(f"🔧 CPU mode con compute_type: {compute_type}")

# Cargar modelo Faster Whisper Large V3
logger.info("🔄 Cargando Faster Whisper large-v3...")
model = WhisperModel(
    "large-v3",
    device=device,
    compute_type=compute_type,
    download_root="/app/models"  # Cache local
)
logger.info("✅ Modelo Faster Whisper large-v3 cargado")


def handler(event):
    try:
        input_data = event.get("input", {})
        audio_url = input_data.get("audio_url")
        language = input_data.get("language", "es")
        task = input_data.get("task", "transcribe")

        if not audio_url:
            return {"error": "audio_url es requerido"}

        # Validar URL para prevenir SSRF
        is_safe, reason = is_safe_url(audio_url)
        if not is_safe:
            logger.warning(f"🚫 URL bloqueada por SSRF: {audio_url} - {reason}")
            return {"error": f"URL no permitida: {reason}"}

        logger.info(f"📥 Descargando: {audio_url}")
        response = requests.get(audio_url, timeout=300, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            temp_path = f.name

        # Opción para word-level timestamps (más lento pero más preciso)
        word_timestamps = input_data.get("word_timestamps", False)

        logger.info(f"🎤 Transcribiendo con Faster Whisper ({device}, {compute_type}, word_ts={word_timestamps})...")

        # Faster Whisper transcribe - retorna generator de segmentos
        segments, info = model.transcribe(
            temp_path,
            language=language,
            task=task,
            beam_size=5,           # Balance entre velocidad y precisión
            best_of=5,             # Mejor precisión
            vad_filter=True,       # Filtrar silencios (más rápido)
            vad_parameters=dict(
                min_silence_duration_ms=500,  # Silencios de 500ms+
            ),
            word_timestamps=word_timestamps  # Habilitar timestamps por palabra si se solicita
        )

        # Convertir generator a lista y construir segments_json con timestamps
        segments_list = list(segments)
        text = " ".join([seg.text.strip() for seg in segments_list])

        # Construir segments_json con todos los timestamps
        # Estructura: [{id, start, end, text, start_ms, end_ms, words?}]
        segments_json = []
        for idx, seg in enumerate(segments_list):
            segment_data = {
                "id": idx,
                "start": round(seg.start, 3),      # Segundos con 3 decimales
                "end": round(seg.end, 3),
                "start_ms": int(seg.start * 1000), # Milisegundos para evidence_snippets
                "end_ms": int(seg.end * 1000),
                "text": seg.text.strip(),
            }

            # Si hay word-level timestamps, incluirlos
            if word_timestamps and hasattr(seg, 'words') and seg.words:
                segment_data["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3) if hasattr(w, 'probability') else None
                    }
                    for w in seg.words
                ]

            segments_json.append(segment_data)

        # Calcular duración
        duration = segments_list[-1].end if segments_list else 0

        if os.path.exists(temp_path):
            os.unlink(temp_path)

        logger.info(f"✅ Transcripción completada: {len(text)} caracteres, {duration:.1f}s, {len(segments_json)} segmentos")
        logger.info(f"📊 Idioma detectado: {info.language} (prob: {info.language_probability:.2f})")

        return {
            "text": text,
            "transcription": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": duration,
            "segments_count": len(segments_json),
            # ======== NUEVO: segments_json para evidence_snippets ========
            "segments_json": segments_json,  # [{id, start, end, start_ms, end_ms, text, words?}]
            "has_word_timestamps": word_timestamps,
            # =============================================================
            "status": "completed",
            "device": device,
            "compute_type": compute_type,
            "model": "faster-whisper-large-v3"
        }

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
