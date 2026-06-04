"""
mixer.py - Audio mixing using FFmpeg

Combines voice audio with ambient sound and musical pad at specified volume levels.
"""

import subprocess
import os
from typing import Optional


def mix_audio(
    voice_path: str,
    ambient_path: str,
    output_path: str,
    voice_volume: float = 1.0,
    ambient_volume: float = 0.2
) -> str:
    """
    Mix voice and ambient audio using FFmpeg.

    Args:
        voice_path: Path to voice audio file.
        ambient_path: Path to ambient audio file.
        output_path: Path to save the mixed audio.
        voice_volume: Voice volume level (0.0 to 1.0, default 1.0).
        ambient_volume: Ambient volume level (0.0 to 1.0, default 0.2).

    Returns:
        Path to the mixed audio file.

    Raises:
        RuntimeError: If FFmpeg is not installed or fails.
    """
    print(f"[Audio] Mixing audio (voice: {voice_volume}, ambient: {ambient_volume})...")

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "FFmpeg not found. Install it from: https://ffmpeg.org/download.html"
        )

    # Check input files exist
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    if not os.path.exists(ambient_path):
        raise FileNotFoundError(f"Ambient file not found: {ambient_path}")

    try:
        # Use FFmpeg to mix audio with volume adjustment
        subprocess.run(
            [
                "ffmpeg",
                "-i", voice_path,
                "-i", ambient_path,
                "-filter_complex",
                f"[0:a]volume={voice_volume}[v];[1:a]volume={ambient_volume}[a];[v][a]amix=inputs=2:duration=first",
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio mixing failed: {e.stderr.decode()}")

    print(f"[Audio] Mixed audio saved to {output_path}")
    return output_path


def mix_audio_3layer(
    voice_path: str,
    ambient_path: str,
    pad_path: str,
    output_path: str,
    voice_volume: float = 1.0,
    ambient_volume: float = 0.15,
    pad_volume: float = 0.1
) -> str:
    """
    Mix voice, ambient, and musical pad audio using FFmpeg.

    Args:
        voice_path: Path to voice audio file.
        ambient_path: Path to ambient audio file.
        pad_path: Path to musical pad audio file.
        output_path: Path to save the mixed audio.
        voice_volume: Voice volume level (0.0 to 1.0, default 1.0).
        ambient_volume: Ambient volume level (0.0 to 1.0, default 0.15).
        pad_volume: Musical pad volume level (0.0 to 1.0, default 0.1).

    Returns:
        Path to the mixed audio file.

    Raises:
        RuntimeError: If FFmpeg is not installed or fails.
    """
    print(f"[Audio] Mixing 3-layer audio (voice: {voice_volume}, ambient: {ambient_volume}, pad: {pad_volume})...")

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "FFmpeg not found. Install it from: https://ffmpeg.org/download.html"
        )

    # Check input files exist
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    if not os.path.exists(ambient_path):
        raise FileNotFoundError(f"Ambient file not found: {ambient_path}")
    if not os.path.exists(pad_path):
        raise FileNotFoundError(f"Pad file not found: {pad_path}")

    try:
        # Use FFmpeg to mix 3 audio layers with volume adjustment
        subprocess.run(
            [
                "ffmpeg",
                "-i", voice_path,
                "-i", ambient_path,
                "-i", pad_path,
                "-filter_complex",
                f"[1:a]volume={ambient_volume}[a1];[2:a]volume={pad_volume}[a2];[0:a][a1][a2]amix=inputs=3:normalize=1",
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio mixing failed: {e.stderr.decode()}")

    print(f"[Audio] Mixed audio saved to {output_path}")
    return output_path


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of an audio file in seconds using FFmpeg.

    Args:
        file_path: Path to audio file.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If FFmpeg fails to get duration.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            capture_output=True,
            check=True,
            text=True
        )
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get audio duration: {e.stderr.decode()}")
