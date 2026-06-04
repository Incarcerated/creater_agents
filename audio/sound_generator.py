"""
sound_generator.py - Ambient sound generation using FFmpeg

Generates loopable ambient sounds (rain, waves, wind) using FFmpeg.
All sounds are generated locally without external files.
"""

import subprocess
import os
from typing import Optional


def generate_ambient_sound(
    output_path: str,
    sound_type: str = "rain",
    duration: Optional[float] = None,
    volume: float = 0.2
) -> str:
    """
    Generate ambient sound using FFmpeg.

    Supports:
    - rain: White noise with low-pass filter
    - waves: Modulated sine waves (ocean-like)
    - wind: Brown noise with modulation

    Args:
        output_path: Path to save the ambient audio (WAV).
        sound_type: Type of sound: "rain", "waves", or "wind".
        duration: Duration in seconds. If None, generates 60 seconds (loopable).
        volume: Volume level (0.0 to 1.0).

    Returns:
        Path to the generated ambient audio file.

    Raises:
        RuntimeError: If FFmpeg is not installed or fails.
        ValueError: If sound_type is invalid.
    """
    print(f"[Audio] Generating ambient sound: {sound_type}...")

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "FFmpeg not found. Install it from: https://ffmpeg.org/download.html"
        )

    # Default duration: 60 seconds (loopable)
    if duration is None:
        duration = 60.0

    # Generate sound based on type
    if sound_type == "rain":
        return _generate_rain(output_path, duration, volume)
    elif sound_type == "waves":
        return _generate_waves(output_path, duration, volume)
    elif sound_type == "wind":
        return _generate_wind(output_path, duration, volume)
    else:
        raise ValueError(f"Invalid sound_type: {sound_type}. Use 'rain', 'waves', or 'wind'")


def _generate_rain(output_path: str, duration: float, volume: float) -> str:
    """
    Generate rain-like sound using white noise with low-pass filter.

    White noise → low-pass filter → rain sound
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"anoisesrc=duration={duration}:color=white:seed=1",
                "-af", f"lowpass=f=600,volume={volume}",  # Lower frequency for softer sound
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Rain sound generation failed: {e.stderr.decode()}")

    return output_path


def _generate_waves(output_path: str, duration: float, volume: float) -> str:
    """
    Generate ocean waves using modulated sine waves.

    Multiple sine waves with different frequencies and modulation.
    """
    try:
        # Combine multiple sine waves for wave effect
        subprocess.run(
            [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"sine=frequency=0.1:duration={duration}",
                "-f", "lavfi",
                "-i", f"sine=frequency=0.15:duration={duration}",
                "-f", "lavfi",
                "-i", f"sine=frequency=0.2:duration={duration}",
                "-filter_complex",
                f"[0:a][1:a]amix=inputs=2:duration=first[a1];[a1][2:a]amix=inputs=2:duration=first,lowpass=f=500,volume={volume}",
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Waves sound generation failed: {e.stderr.decode()}")

    return output_path


def _generate_wind(output_path: str, duration: float, volume: float) -> str:
    """
    Generate wind-like sound using brown noise with modulation.

    Brown noise → modulation → wind sound
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"anoisesrc=duration={duration}:color=brown:seed=2",
                "-af", f"lowpass=f=400,volume={volume}",  # Add lowpass for softer wind
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Wind sound generation failed: {e.stderr.decode()}")

    return output_path


def _generate_lullaby_pad(output_path: str, duration: float, volume: float) -> str:
    """
    Generate soft lullaby pad using C major chord sine waves.

    C major chord: C (261.6 Hz), E (329.6 Hz), G (392 Hz)
    Mixes three sine waves with lowpass filter for soft tone.

    Args:
        output_path: Path to save the pad audio.
        duration: Duration in seconds.
        volume: Volume level (0.0 to 1.0).

    Returns:
        Path to the generated pad audio file.
    """
    try:
        # Generate C major chord using three sine waves
        subprocess.run(
            [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"sine=frequency=261.6:duration={duration}",  # C4
                "-f", "lavfi",
                "-i", f"sine=frequency=329.6:duration={duration}",  # E4
                "-f", "lavfi",
                "-i", f"sine=frequency=392.0:duration={duration}",  # G4
                "-filter_complex",
                f"[0:a][1:a][2:a]amix=inputs=3:duration=first,lowpass=f=800,volume={volume}",
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Lullaby pad generation failed: {e.stderr.decode()}")

    return output_path


def extend_ambient_to_match(
    ambient_path: str,
    target_duration: float,
    output_path: str
) -> str:
    """
    Loop ambient sound to match target duration.

    Args:
        ambient_path: Path to source ambient audio.
        target_duration: Target duration in seconds.
        output_path: Path to save the extended audio.

    Returns:
        Path to the extended ambient audio file.
    """
    print(f"[Audio] Extending ambient to {target_duration} seconds...")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-stream_loop", "-1",  # Loop infinitely
                "-i", ambient_path,
                "-t", str(target_duration),
                "-c", "copy",
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ambient extension failed: {e.stderr.decode()}")

    return output_path
