"""
audio_pipeline.py - Orchestrates the full audio generation pipeline

Coordinates TTS, ambient sound generation, lullaby pad, and mixing to produce final audio.
Includes caching to avoid regenerating the same script.
"""

import os
import hashlib
from typing import Optional, Dict
from audio.tts import generate_voice
from audio.sound_generator import generate_ambient_sound, extend_ambient_to_match, _generate_lullaby_pad
from audio.mixer import mix_audio_3layer, get_audio_duration


def _parse_sound_profile(sound_profile: str) -> tuple:
    """
    Parse sound profile to extract ambient type.

    Args:
        sound_profile: Sound profile (e.g., "rain_lullaby", "ocean_lullaby", "wind_lullaby").

    Returns:
        Tuple of (ambient_type, profile_name).
    """
    if sound_profile.endswith("_lullaby"):
        ambient_type = sound_profile.replace("_lullaby", "")
        if ambient_type == "ocean":
            ambient_type = "waves"
        return ambient_type, sound_profile
    return sound_profile, sound_profile


def generate_audio_for_script(
    script: Dict,
    output_dir: str,
    sound_profile: str = "rain_lullaby",
    voice_volume: float = 1.0,
    ambient_volume: float = 0.15,
    pad_volume: float = 0.1,
    quality_mode: str = "quality",
    voice_preset: str = "v2/en_speaker_6",
    use_cache: bool = True
) -> str:
    """
    Generate complete audio for a single script with 3-layer mixing.

    Pipeline:
    1. Check cache (if enabled)
    2. Generate voice from lyrics (TTS with softening)
    3. Generate ambient sound
    4. Generate lullaby pad (musical layer)
    5. Extend ambient and pad to match voice duration
    6. Mix voice + ambient + pad
    7. Save final audio

    Args:
        script: Script dict with 'lyrics', 'title', 'idea_id'.
        output_dir: Directory to save output audio.
        sound_profile: Sound profile: "rain_lullaby", "ocean_lullaby", "wind_lullaby".
        voice_volume: Voice volume level (0.0 to 1.0).
        ambient_volume: Ambient volume level (0.0 to 1.0).
        pad_volume: Musical pad volume level (0.0 to 1.0).
        quality_mode: "quality" (Bark) or "fast" (macOS say).
        voice_preset: Bark voice preset.
        use_cache: Whether to check cache before generating.

    Returns:
        Path to the final mixed audio file.

    Raises:
        RuntimeError: If any step fails.
    """
    title = script.get("title", "unknown")
    lyrics = script.get("lyrics", "")
    idea_id = script.get("idea_id", "unknown")

    print(f"\n🎵 Audio Pipeline: Generating audio for '{title}' (idea #{idea_id})...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse sound profile
    ambient_type, profile_name = _parse_sound_profile(sound_profile)

    # Generate cache key from lyrics + sound_profile + volumes + quality_mode + voice_preset
    cache_key = _generate_cache_key(lyrics, sound_profile, voice_volume, ambient_volume, pad_volume, quality_mode, voice_preset)
    cache_file = os.path.join(output_dir, f"{cache_key}.wav")

    # Check cache
    if use_cache and os.path.exists(cache_file):
        print(f"[Audio] Cache hit: {cache_file}")
        return cache_file

    # Step 1: Generate voice (with softening)
    print(f"[Audio] Step 1: Generating voice...")
    voice_file = os.path.join(output_dir, f"{idea_id}_voice.wav")
    generate_voice(lyrics, voice_file, soften=True, quality_mode=quality_mode, voice_preset=voice_preset)

    # Step 2: Get voice duration
    voice_duration = get_audio_duration(voice_file)
    print(f"[Audio] Voice duration: {voice_duration:.2f} seconds")

    # Step 3: Generate ambient sound
    print(f"[Audio] Step 2: Generating ambient sound ({ambient_type})...")
    ambient_file = os.path.join(output_dir, f"{idea_id}_ambient.wav")
    generate_ambient_sound(ambient_file, sound_type=ambient_type, duration=60.0, volume=ambient_volume)

    # Step 4: Generate lullaby pad
    print(f"[Audio] Step 3: Generating lullaby pad...")
    pad_file = os.path.join(output_dir, f"{idea_id}_pad.wav")
    _generate_lullaby_pad(pad_file, duration=60.0, volume=pad_volume)

    # Step 5: Extend ambient and pad to match voice duration
    print(f"[Audio] Step 4: Extending ambient and pad to match voice duration...")
    ambient_extended = os.path.join(output_dir, f"{idea_id}_ambient_extended.wav")
    pad_extended = os.path.join(output_dir, f"{idea_id}_pad_extended.wav")
    extend_ambient_to_match(ambient_file, voice_duration, ambient_extended)
    extend_ambient_to_match(pad_file, voice_duration, pad_extended)

    # Step 6: Mix voice + ambient + pad
    print(f"[Audio] Step 5: Mixing 3-layer audio...")
    final_output = os.path.join(output_dir, f"{idea_id}_{title.replace(' ', '_')}.wav")
    mix_audio_3layer(voice_file, ambient_extended, pad_extended, final_output, voice_volume, ambient_volume, pad_volume)

    # Step 7: Cleanup intermediate files
    for f in [voice_file, ambient_file, pad_file, ambient_extended, pad_extended]:
        if os.path.exists(f):
            os.remove(f)

    # Step 8: Save to cache if enabled
    if use_cache:
        os.rename(final_output, cache_file)
        print(f"[Audio] Saved to cache: {cache_file}")
        return cache_file

    print(f"[Audio] Final audio saved to {final_output}")
    return final_output


def generate_audio_for_scripts(
    scripts: list[Dict],
    output_dir: str,
    sound_profile: str = "rain_lullaby",
    voice_volume: float = 1.0,
    ambient_volume: float = 0.15,
    pad_volume: float = 0.1,
    quality_mode: str = "quality",
    voice_preset: str = "v2/en_speaker_6",
    use_cache: bool = True
) -> list[str]:
    """
    Generate audio for multiple scripts.

    Args:
        scripts: List of script dicts.
        output_dir: Directory to save output audio.
        sound_profile: Sound profile.
        voice_volume: Voice volume level.
        ambient_volume: Ambient volume level.
        pad_volume: Musical pad volume level.
        quality_mode: Quality mode (quality/fast).
        voice_preset: Voice preset.
        use_cache: Whether to use cache.

    Returns:
        List of paths to generated audio files.
    """
    print(f"\n🎵 Audio Pipeline: Processing {len(scripts)} scripts...")

    audio_files = []
    for script in scripts:
        try:
            audio_path = generate_audio_for_script(
                script,
                output_dir,
                sound_profile,
                voice_volume,
                ambient_volume,
                pad_volume,
                quality_mode,
                voice_preset,
                use_cache
            )
            audio_files.append(audio_path)
        except Exception as e:
            print(f"[Audio] Failed to generate audio for script #{script.get('idea_id', '?')}: {e}")
            continue

    print(f"\n✅ Audio Pipeline: Generated {len(audio_files)} audio files")
    return audio_files


def _generate_cache_key(
    lyrics: str,
    sound_profile: str,
    voice_volume: float,
    ambient_volume: float,
    pad_volume: float,
    quality_mode: str = "quality",
    voice_preset: str = "v2/en_speaker_6"
) -> str:
    """
    Generate a unique cache key from input parameters.

    Args:
        lyrics: The lyrics text.
        sound_profile: Sound profile.
        voice_volume: Voice volume.
        ambient_volume: Ambient volume.
        pad_volume: Pad volume.
        quality_mode: Quality mode (quality/fast).
        voice_preset: Voice preset.

    Returns:
        MD5 hash as cache key.
    """
    cache_string = f"{lyrics}|{sound_profile}|{voice_volume}|{ambient_volume}|{pad_volume}|{quality_mode}|{voice_preset}"
    return hashlib.md5(cache_string.encode()).hexdigest()


def clear_cache(output_dir: str) -> None:
    """
    Clear all cached audio files in the output directory.

    Args:
        output_dir: Directory containing cached files.
    """
    if not os.path.exists(output_dir):
        return

    for filename in os.listdir(output_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(output_dir, filename)
            os.remove(file_path)
            print(f"[Audio] Cleared cache: {file_path}")
