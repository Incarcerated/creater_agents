#!/usr/bin/env python3
"""
generate_audio.py - Generate audio for a single script

Usage:
    python generate_audio.py <script_json_file> [sound_type]

Example:
    python generate_audio.py memory/script/2026-04-30_00-37-16_whispering_waves_lullaby.json rain
"""

import sys
import os
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio import generate_audio_for_script
from config import get_config


def main():
    parser = argparse.ArgumentParser(description="Generate audio for a single script")
    parser.add_argument("script_file", help="Path to script JSON file")
    parser.add_argument("sound_profile", nargs="?", default="rain_lullaby", 
                       choices=["rain_lullaby", "ocean_lullaby", "wind_lullaby"], help="Sound profile")
    parser.add_argument("--voice-volume", type=float, default=1.0, help="Voice volume (0.0-1.0)")
    parser.add_argument("--ambient-volume", type=float, default=0.15, help="Ambient volume (0.0-1.0)")
    parser.add_argument("--pad-volume", type=float, default=0.1, help="Pad volume (0.0-1.0)")
    parser.add_argument("--quality-mode", choices=["quality", "fast"], default="quality", 
                       help="Quality mode: 'quality' (Bark TTS) or 'fast' (macOS say)")
    parser.add_argument("--voice-preset", default="v2/en_speaker_6", 
                       help="Bark voice preset (e.g., v2/en_speaker_6)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    # Load script file
    if not os.path.exists(args.script_file):
        print(f"Error: Script file not found: {args.script_file}")
        sys.exit(1)

    with open(args.script_file, "r") as f:
        data = json.load(f)

    # Extract script (handle both single script and scripts array)
    if "scripts" in data and isinstance(data["scripts"], list):
        script = data["scripts"][0]  # Use first script
    elif "lyrics" in data:
        script = data  # Already a single script
    else:
        print("Error: Invalid script format. Expected 'scripts' array or 'lyrics' field.")
        sys.exit(1)

    # Get config
    config = get_config()

    # Generate audio
    output_dir = os.path.join(os.path.dirname(__file__), "output", "audio")
    
    try:
        audio_path = generate_audio_for_script(
            script,
            output_dir,
            sound_profile=args.sound_profile,
            voice_volume=args.voice_volume,
            ambient_volume=args.ambient_volume,
            pad_volume=args.pad_volume,
            quality_mode=args.quality_mode,
            voice_preset=args.voice_preset,
            use_cache=not args.no_cache
        )
        print(f"\n✅ Audio generated: {audio_path}")
    except Exception as e:
        print(f"\n❌ Error generating audio: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
