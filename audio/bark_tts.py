"""
bark_tts.py - Text-to-Speech using Bark (Suno AI)

Generates expressive, human-like audio using Bark's neural TTS model.
Supports voice presets and hardware acceleration for Apple Silicon.
"""

import os
from typing import Optional

# Optional imports - Bark may not be installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Global model cache to avoid reloading
_bark_model_loaded = False
_device = None


def _get_device() -> str:
    """
    Detect and return the best available device for Bark.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    global _device
    if _device is not None:
        return _device

    if not TORCH_AVAILABLE:
        _device = "cpu"
        print(f"[Bark] Torch not available, using CPU")
        return _device

    if torch.cuda.is_available():
        _device = "cuda"
        print(f"[Bark] Using CUDA device")
    elif torch.backends.mps.is_available():
        _device = "mps"
        print(f"[Bark] Using MPS (Apple Silicon) device")
    else:
        _device = "cpu"
        print(f"[Bark] Using CPU device")
    
    return _device


def _load_bark():
    """
    Load Bark model (cached globally for performance).
    """
    global _bark_model_loaded
    
    if _bark_model_loaded:
        return
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch not installed. Cannot use Bark TTS.")
    
    try:
        from bark import generate_audio, preload_models
        print(f"[Bark] Loading Bark models on {_get_device()}...")
        
        # Handle PyTorch 2.6+ weights_only change
        # Monkey-patch torch.load to disable weights_only for Bark
        original_torch_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_torch_load(*args, **kwargs)
        torch.load = patched_load
        
        try:
            # Preload models to device
            preload_models(
                text_use_gpu=True if _get_device() != "cpu" else False,
                fine_use_gpu=True if _get_device() != "cpu" else False,
                coarse_use_gpu=True if _get_device() != "cpu" else False,
                codec_use_gpu=True if _get_device() != "cpu" else False,
            )
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
        
        _bark_model_loaded = True
        print(f"[Bark] Models loaded successfully")
    except ImportError:
        raise RuntimeError(
            "Bark not installed. Install it with:\n"
            "pip install git+https://github.com/suno-ai/bark.git\n"
            "pip install torch torchaudio scipy"
        )


def preprocess_text_for_bark(text: str) -> str:
    """
    Preprocess text for Bark TTS.

    - Keep line breaks for natural pauses
    - Add ellipsis for longer pauses
    - Keep sentences short for better prosody

    Args:
        text: Original lyrics text.

    Returns:
        Preprocessed text optimized for Bark.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # Add pauses between lines using ellipsis
    processed = " ... ".join(lines)
    
    # Ensure text isn't too long (Bark works best with shorter segments)
    if len(processed) > 500:
        # Split into chunks if too long
        words = processed.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > 250:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        processed = " ... ".join(chunks)
    
    return processed


def generate_bark_audio(
    text: str,
    output_path: str,
    voice_preset: str = "v2/en_speaker_6",
    history_prompt: Optional[str] = None
) -> str:
    """
    Generate audio using Bark TTS.

    Args:
        text: Text to convert to speech.
        output_path: Path to save the generated audio (WAV).
        voice_preset: Bark voice preset (default: v2/en_speaker_6 - soft female).
        history_prompt: Optional history prompt for continuity.

    Returns:
        Path to the generated audio file.

    Raises:
        RuntimeError: If Bark fails to generate audio.
    """
    print(f"[Audio] Generating Bark audio with voice preset: {voice_preset}...")
    
    # Load Bark model (cached)
    _load_bark()
    
    # Preprocess text
    processed_text = preprocess_text_for_bark(text)
    print(f"[Audio] Text preprocessed for Bark")
    
    try:
        from bark import generate_audio, SAMPLE_RATE
        from scipy.io.wavfile import write as write_wav
        
        # Generate audio
        print(f"[Audio] Bark generating audio...")
        audio_array = generate_audio(
            processed_text,
            history_prompt=voice_preset,
            silent=True
        )
        
        # Save as WAV
        write_wav(output_path, SAMPLE_RATE, audio_array)
        
        print(f"[Audio] Bark generation complete: {output_path}")
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Bark TTS failed: {e}")


def is_bark_available() -> bool:
    """
    Check if Bark is installed and available.

    Returns:
        True if Bark is available, False otherwise.
    """
    try:
        import bark
        return True
    except ImportError:
        return False


def get_available_voice_presets() -> list:
    """
    Get list of available Bark voice presets.

    Returns:
        List of voice preset strings.
    """
    # Common Bark voice presets
    presets = [
        "v2/en_speaker_0",  # Neutral male
        "v2/en_speaker_1",  # Neutral female
        "v2/en_speaker_2",  # Calm male
        "v2/en_speaker_3",  # Calm neutral
        "v2/en_speaker_4",  # Soft male
        "v2/en_speaker_5",  # Soft female
        "v2/en_speaker_6",  # Soft female (recommended for lullabies)
        "v2/en_speaker_7",  # Gentle male
        "v2/en_speaker_8",  # Gentle female
        "v2/en_speaker_9",  # Deeper tone
    ]
    return presets
