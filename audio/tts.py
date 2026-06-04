"""
tts.py - Text-to-Speech using Bark (primary) with macOS 'say' fallback

Converts lyrics to voice audio using Bark neural TTS for quality,
with automatic fallback to macOS 'say' for speed or if Bark fails.
"""

import os
import subprocess
import tempfile
from typing import Optional

from audio.bark_tts import generate_bark_audio, is_bark_available


def preprocess_lyrics(lyrics: str) -> str:
    """
    Pre-process lyrics for TTS by adding pauses between lines.

    Args:
        lyrics: Original lyrics text.

    Returns:
        Lyrics with pauses added between lines.
    """
    lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
    # Add pauses between lines using ellipsis
    processed = " ... ".join(lines)
    return processed


def generate_voice(
    lyrics: str,
    output_path: str,
    voice: str = "Samantha",
    speed: float = 0.9,
    chunk_size: int = 500,
    soften: bool = True,
    quality_mode: str = "quality",
    voice_preset: str = "v2/en_speaker_6"
) -> str:
    """
    Generate voice audio from lyrics using Bark (quality) or macOS 'say' (fast).

    Args:
        lyrics: The lyrics text to convert to speech.
        output_path: Path to save the final voice audio (WAV).
        voice: macOS voice name (fallback only, default: Samantha).
        speed: Speech speed for fallback (0.5 to 2.0, default 0.9).
        chunk_size: Number of characters per chunk for fallback (default: 500).
        soften: Whether to apply voice softening post-processing.
        quality_mode: "quality" (Bark) or "fast" (macOS say).
        voice_preset: Bark voice preset (default: v2/en_speaker_6).

    Returns:
        Path to the generated voice audio file.

    Raises:
        RuntimeError: If both Bark and fallback fail.
    """
    # Try Bark first if quality mode
    if quality_mode == "quality" and is_bark_available():
        try:
            print(f"[Audio] Using Bark TTS (quality mode)...")
            generate_bark_audio(lyrics, output_path, voice_preset=voice_preset)
            
            # Apply light post-processing for Bark (less aggressive than say)
            if soften:
                print(f"[Audio] Applying light processing for Bark...")
                softened_path = output_path.replace(".wav", "_soft.wav")
                soften_voice_light(output_path, softened_path)
                os.remove(output_path)
                os.rename(softened_path, output_path)
            
            print(f"[Audio] Voice audio saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"[Audio] Bark failed: {e}")
            print(f"[Audio] Falling back to macOS 'say'...")
    
    # Fallback to macOS 'say'
    print(f"[Audio] Using macOS 'say' command (fast mode)...")
    return generate_voice_say(lyrics, output_path, voice, speed, chunk_size, soften)


def generate_voice_say(
    lyrics: str,
    output_path: str,
    voice: str = "Samantha",
    speed: float = 0.9,
    chunk_size: int = 500,
    soften: bool = True
) -> str:
    """
    Generate voice audio using macOS 'say' command (fallback).

    Args:
        lyrics: The lyrics text to convert to speech.
        output_path: Path to save the final voice audio (WAV).
        voice: macOS voice name (default: Samantha).
        speed: Speech speed (0.5 to 2.0, default 0.9 for lullabies).
        chunk_size: Number of characters per chunk (default: 500).
        soften: Whether to apply voice softening post-processing.

    Returns:
        Path to the generated voice audio file.

    Raises:
        RuntimeError: If 'say' command is not available or fails.
    """
    # Check if say is available (macOS only)
    if not os.path.exists("/usr/bin/say"):
        raise RuntimeError(
            "macOS 'say' command not found. This TTS module only works on macOS."
        )

    # Pre-process lyrics with pauses
    processed_lyrics = preprocess_lyrics(lyrics)

    # Split lyrics into chunks
    lines = [line.strip() for line in processed_lyrics.split("\n") if line.strip()]
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += " " + line
            else:
                current_chunk = line

    if current_chunk:
        chunks.append(current_chunk)

    print(f"[Audio] Split lyrics into {len(chunks)} chunks")

    # Generate audio for each chunk
    chunk_files = []
    temp_dir = tempfile.mkdtemp()

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.aiff")
        print(f"[Audio] Generating chunk {i+1}/{len(chunks)}...")

        try:
            # Use macOS say command to generate audio
            subprocess.run(
                [
                    "say",
                    "-v", voice,
                    "-r", str(int(160 * speed)),  # Speech rate
                    "-o", chunk_path,
                    chunk
                ],
                capture_output=True,
                check=True
            )
            chunk_files.append(chunk_path)
        except subprocess.CalledProcessError as e:
            print(f"[Audio] Warning: Chunk {i+1} failed: {e}")
            # Create empty file to maintain order
            with open(chunk_path, "w") as f:
                f.write("")

    # Merge chunks using FFmpeg
    if chunk_files:
        print(f"[Audio] Merging {len(chunk_files)} chunks...")
        merge_audio_files(chunk_files, output_path)
    else:
        raise RuntimeError("No audio chunks were generated")

    # Cleanup temp files
    for f in chunk_files:
        if os.path.exists(f):
            os.remove(f)
    os.rmdir(temp_dir)

    # Apply voice softening if enabled (aggressive for say)
    if soften:
        print(f"[Audio] Applying voice softening...")
        softened_path = output_path.replace(".wav", "_soft.wav")
        soften_voice(output_path, softened_path)
        # Replace original with softened
        os.remove(output_path)
        os.rename(softened_path, output_path)

    print(f"[Audio] Voice audio saved to {output_path}")
    return output_path


def merge_audio_files(input_files: list, output_path: str) -> None:
    """
    Merge multiple audio files into one using FFmpeg concat demuxer.

    Args:
        input_files: List of input audio file paths.
        output_path: Path to save the merged audio.

    Raises:
        RuntimeError: If FFmpeg is not installed or fails.
    """
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "FFmpeg not found. Install it from: https://ffmpeg.org/download.html"
        )

    # Create concat file list
    temp_dir = tempfile.mkdtemp()
    concat_file = os.path.join(temp_dir, "concat.txt")

    with open(concat_file, "w") as f:
        for input_file in input_files:
            # Escape file paths for FFmpeg
            escaped_path = input_file.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    # Use FFmpeg to concat and convert to WAV
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c:a", "pcm_s16le",  # Convert to little-endian PCM (WAV compatible)
                "-ar", "22050",      # Set sample rate
                "-ac", "1",           # Mono
                output_path,
                "-y"  # Overwrite output file if exists
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg concat failed: {e.stderr.decode()}")

    # Cleanup
    os.remove(concat_file)
    os.rmdir(temp_dir)


def soften_voice(input_path: str, output_path: str) -> None:
    """
    Apply voice softening effects using FFmpeg (aggressive for macOS say).

    Effects:
    - Slower speech (atempo=0.85)
    - Lower pitch (asetrate=44100*0.9)
    - Lowpass filter for softer tone (lowpass=f=1200)
    - Volume boost (volume=1.2)

    Args:
        input_path: Path to input voice audio.
        output_path: Path to save softened audio.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", input_path,
                "-af", "atempo=0.85,asetrate=44100*0.9,lowpass=f=1200,volume=1.2",
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Voice softening failed: {e.stderr.decode()}")


def soften_voice_light(input_path: str, output_path: str) -> None:
    """
    Apply light voice processing for Bark (less aggressive).

    Bark already has natural tone, so we only:
    - Slight volume boost (volume=1.1)
    - Mild lowpass filter (lowpass=f=3000)

    Args:
        input_path: Path to input voice audio.
        output_path: Path to save processed audio.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", input_path,
                "-af", "volume=1.1,lowpass=f=3000",
                "-y", output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Voice processing failed: {e.stderr.decode()}")
