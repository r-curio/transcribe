#!/usr/bin/env python3
"""
Example usage of the AudioTranscriber class
"""

from transcribe import AudioTranscriber
import os

def main():
    # Example audio file path (replace with your actual audio file)
    audio_file = "example.wav"  # Change this to your audio file path
    
    if not os.path.exists(audio_file):
        print(f"Please provide a valid audio file path. Current path '{audio_file}' doesn't exist.")
        print("You can either:")
        print("1. Place your audio file in the current directory and update the path in this script")
        print("2. Use the command line interface: python transcribe.py your_audio_file.wav")
        return
    
    # Initialize transcriber with the same model from your main.py
    model_name = "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official"
    transcriber = AudioTranscriber(model_name)
    
    # Perform transcription with 20-second chunks for better timestamp accuracy
    print(f"Transcribing {audio_file}...")
    transcriptions = transcriber.transcribe_with_timestamps(audio_file, chunk_length_s=20.0)
    
    if transcriptions:
        print(f"\nTranscription completed! Found {len(transcriptions)} segments:")
        
        # Print first few segments as preview
        for i, segment in enumerate(transcriptions[:3]):
            start_time = transcriber.format_timestamps(segment["start_time"])
            end_time = transcriber.format_timestamps(segment["end_time"])
            print(f"Segment {i+1}: [{start_time} - {end_time}] {segment['text']}")
        
        if len(transcriptions) > 3:
            print(f"... and {len(transcriptions) - 3} more segments")
        
        # Save in different formats
        base_name = os.path.splitext(audio_file)[0]
        
        # Save as text
        transcriber.save_transcription(transcriptions, f"{base_name}_transcription.txt", "txt")
        
        # Save as JSON for programmatic use
        transcriber.save_transcription(transcriptions, f"{base_name}_transcription.json", "json")
        
        # Save as SRT for video subtitles
        transcriber.save_transcription(transcriptions, f"{base_name}_transcription.srt", "srt")
        
        print(f"\nTranscription files saved:")
        print(f"- {base_name}_transcription.txt (human-readable)")
        print(f"- {base_name}_transcription.json (structured data)")
        print(f"- {base_name}_transcription.srt (subtitle format)")
    else:
        print("No transcription generated. The audio might be silent or too short.")

if __name__ == "__main__":
    main() 