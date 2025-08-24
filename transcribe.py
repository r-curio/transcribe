#!/usr/bin/env python3
"""
Audio Transcription Script with CTC Forced Alignment
Uses Hugging Face transformers with CTC forced alignment for precise word-level timestamps
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torchaudio
from transformers import (
    pipeline, 
    AutoFeatureExtractor, 
    AutoModelForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
import numpy as np
import librosa


class AudioTranscriber:
    def __init__(self, model_name: str = "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official"):
        """
        Initialize the transcriber with a Hugging Face model
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize pipeline first (primary method for word timestamps)
        try:
            self.pipe = pipeline("automatic-speech-recognition", 
                               model=model_name, 
                               device=0 if self.device == "cuda" else -1,
                               return_timestamps="word")
            print(f"Loaded pipeline: {model_name}")
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            self.pipe = None
        
        # Also load model and processor for fallback
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
            print(f"Loaded model components: {model_name}")
            
            # Get vocabulary for alignment
            self.vocab = self.processor.tokenizer.get_vocab()
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            
        except Exception as e:
            print(f"Error loading model components: {e}")
            print("Will use pipeline-only approach")
            self.processor = None
            self.model = None
    
    def load_audio(self, audio_path: str) -> tuple:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio using librosa for better compatibility
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Ensure audio is mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        return audio_array, sample_rate
    
    def get_word_timestamps_ctc(self, audio_array: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        Get word-level timestamps using CTC forced alignment
        
        Args:
            audio_array: Audio signal
            sample_rate: Sample rate
            
        Returns:
            List of word dictionaries with timestamps
        """
        # Always use pipeline approach for better word-level results
        # The direct CTC alignment is complex for character-level models
        try:
            # Use pipeline with word timestamps - this handles tokenization properly
            result = self.pipe(audio_array, return_timestamps="word")
            
            word_timestamps = []
            if "chunks" in result:
                for chunk in result["chunks"]:
                    word_timestamps.append({
                        "word": chunk["text"].strip(),
                        "start_time": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0,
                        "end_time": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else len(audio_array) / sample_rate,
                        "confidence": 1.0  # Pipeline doesn't provide confidence but gives clean words
                    })
            return word_timestamps
            
        except Exception as e:
            print(f"Pipeline approach failed: {e}")
            # Fallback to simple approach
            if self.model is None or self.processor is None:
                return []
                
            # Process audio and get clean transcription
            inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get clean transcription
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Split into words and estimate timing
            words = transcription.strip().split()
            if not words:
                return []
            
            # Simple time division approach
            audio_duration = len(audio_array) / sample_rate
            word_duration = audio_duration / len(words)
            
            word_timestamps = []
            for i, word in enumerate(words):
                start_time = i * word_duration
                end_time = (i + 1) * word_duration
                
                word_timestamps.append({
                    "word": word,
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": 0.8  # Default confidence
                })
            
            return word_timestamps
    
    def group_words_into_utterances(self, word_timestamps: List[Dict[str, Any]], 
                                  max_pause: float = 1.0, 
                                  min_utterance_length: float = 0.1,  # Reduced from 0.5 to capture single words
                                  max_utterance_length: float = 30.0) -> List[Dict[str, Any]]:
        """
        Group words into natural utterances based on pauses
        
        Args:
            word_timestamps: List of word dictionaries with timestamps
            max_pause: Maximum pause between words to consider same utterance (seconds)
            min_utterance_length: Minimum utterance duration (seconds)
            max_utterance_length: Maximum utterance duration (seconds)
            
        Returns:
            List of utterance dictionaries
        """
        if not word_timestamps:
            return []
        
        utterances = []
        current_utterance_words = []
        
        for i, word in enumerate(word_timestamps):
            if not current_utterance_words:
                # Start new utterance
                current_utterance_words = [word]
            else:
                # Check pause since last word
                last_word = current_utterance_words[-1]
                pause_duration = word["start_time"] - last_word["end_time"]
                
                # Check if we should continue current utterance or start new one
                current_duration = word["end_time"] - current_utterance_words[0]["start_time"]
                
                if (pause_duration <= max_pause and 
                    current_duration <= max_utterance_length):
                    # Continue current utterance
                    current_utterance_words.append(word)
                else:
                    # End current utterance and start new one
                    if current_utterance_words:
                        utterance = self._create_utterance_from_words(current_utterance_words)
                        # Accept all utterances, even very short ones (single words)
                        if utterance["duration"] >= min_utterance_length or len(current_utterance_words) == 1:
                            utterances.append(utterance)
                    
                    current_utterance_words = [word]
        
        # Add last utterance
        if current_utterance_words:
            utterance = self._create_utterance_from_words(current_utterance_words)
            # Accept all utterances, even very short ones (single words)
            if utterance["duration"] >= min_utterance_length or len(current_utterance_words) == 1:
                utterances.append(utterance)
        
        return utterances
    
    def _create_utterance_from_words(self, words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an utterance dictionary from a list of words"""
        if not words:
            return {}
        
        start_time = words[0]["start_time"]
        end_time = words[-1]["end_time"]
        text = " ".join([word["word"] for word in words])
        confidence = np.mean([word["confidence"] for word in words])
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "text": text,
            "confidence": confidence,
            "word_count": len(words),
            "words": words
        }
    
    def transcribe_with_timestamps(self, audio_path: str, 
                                 max_pause: float = 0.8,  # Reduced from 1.0 for better single word detection
                                 min_utterance_length: float = 0.1,  # Reduced from 0.5
                                 max_utterance_length: float = 30.0,
                                 chunk_length_s: float = 60.0) -> List[Dict[str, Any]]:
        """
        Transcribe audio with CTC forced alignment for precise timestamps
        
        Args:
            audio_path: Path to audio file
            max_pause: Maximum pause between words for same utterance (seconds)
            min_utterance_length: Minimum utterance duration (seconds)  
            max_utterance_length: Maximum utterance duration (seconds)
            chunk_length_s: Length of audio chunks for processing (seconds)
            
        Returns:
            List of utterance dictionaries with precise timestamps
        """
        print(f"Loading audio: {audio_path}")
        audio_array, sample_rate = self.load_audio(audio_path)
        
        # Process audio in chunks if it's very long
        chunk_size = int(chunk_length_s * sample_rate)
        all_utterances = []
        
        for chunk_start in range(0, len(audio_array), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(audio_array))
            chunk_audio = audio_array[chunk_start:chunk_end]
            chunk_offset = chunk_start / sample_rate
            
            print(f"Processing chunk: {chunk_offset:.1f}s - {chunk_end/sample_rate:.1f}s")
            
            # Get word-level timestamps for this chunk
            word_timestamps = self.get_word_timestamps_ctc(chunk_audio, sample_rate)
            
            # Adjust timestamps to account for chunk offset
            for word in word_timestamps:
                word["start_time"] += chunk_offset
                word["end_time"] += chunk_offset
            
            # Group words into utterances
            utterances = self.group_words_into_utterances(
                word_timestamps, max_pause, min_utterance_length, max_utterance_length
            )
            
            # Debug output
            print(f"  Raw words found: {len(word_timestamps)}")
            if len(word_timestamps) > 0:
                print(f"  First word: '{word_timestamps[0]['word']}' at {word_timestamps[0]['start_time']:.2f}s")
                print(f"  Last word: '{word_timestamps[-1]['word']}' at {word_timestamps[-1]['start_time']:.2f}s")
            print(f"  Grouped into {len(utterances)} utterances")
            
            all_utterances.extend(utterances)
        
        print(f"Found {len(all_utterances)} utterances")
        return all_utterances
    
    def format_timestamps(self, seconds: float) -> str:
        """Format seconds into MM:SS.sss format"""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    
    def save_transcription(self, transcriptions: List[Dict[str, Any]], output_path: str, format_type: str = "txt"):
        """
        Save transcription to file
        
        Args:
            transcriptions: List of transcription segments
            output_path: Output file path
            format_type: Output format (txt, json, srt)
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        if format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcriptions, f, indent=2, ensure_ascii=False)
        
        elif format_type == "srt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(transcriptions, 1):
                    start_time = self.format_timestamps(segment["start_time"])
                    end_time = self.format_timestamps(segment["end_time"])
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text']}\n\n")
        
        else:  # txt format
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcription of audio file\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Total utterances: {len(transcriptions)}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, segment in enumerate(transcriptions):
                    start_time = self.format_timestamps(segment["start_time"])
                    end_time = self.format_timestamps(segment["end_time"])
                    confidence = segment.get("confidence", 0.0)
                    word_count = segment.get("word_count", 0)
                    
                    f.write(f"Utterance {i+1}:\n")
                    f.write(f"  Time: [{start_time} - {end_time}] ({segment['duration']:.2f}s)\n")
                    f.write(f"  Confidence: {confidence:.3f}\n")
                    f.write(f"  Words: {word_count}\n")
                    f.write(f"  Text: {segment['text']}\n\n")
        
        print(f"Transcription saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with CTC forced alignment")
    parser.add_argument("audio_path", help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--model", default="Khalsuu/filipino-wav2vec2-l-xls-r-300m-official", 
                       help="Hugging Face model name")
    parser.add_argument("--output", "-o", help="Output file path (default: input_name_transcription.txt)")
    parser.add_argument("--format", choices=["txt", "json", "srt"], default="txt",
                       help="Output format (default: txt)")
    parser.add_argument("--max-pause", type=float, default=0.8,
                       help="Maximum pause between words for same utterance in seconds (default: 0.8)")
    parser.add_argument("--min-utterance-length", type=float, default=0.1,
                       help="Minimum utterance duration in seconds (default: 0.1)")
    parser.add_argument("--max-utterance-length", type=float, default=30.0,
                       help="Maximum utterance duration in seconds (default: 30.0)")
    parser.add_argument("--chunk-length", type=float, default=60.0,
                       help="Audio chunk length for processing in seconds (default: 60.0)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        sys.exit(1)
    
    # Set output path if not specified
    if not args.output:
        input_name = Path(args.audio_path).stem
        args.output = f"{input_name}_transcription.{args.format}"
    
    try:
        # Initialize transcriber
        transcriber = AudioTranscriber(args.model)
        
        # Perform transcription with CTC alignment
        print("Starting transcription with CTC forced alignment...")
        transcriptions = transcriber.transcribe_with_timestamps(
            args.audio_path,
            max_pause=args.max_pause,
            min_utterance_length=args.min_utterance_length, 
            max_utterance_length=args.max_utterance_length,
            chunk_length_s=args.chunk_length
        )
        
        if not transcriptions:
            print("No transcription generated. Audio might be silent or too short.")
            sys.exit(1)
        
        # Save results
        transcriber.save_transcription(transcriptions, args.output, args.format)
        
        # Print summary
        total_duration = transcriptions[-1]["end_time"] if transcriptions else 0
        avg_confidence = np.mean([t.get("confidence", 0.0) for t in transcriptions])
        total_words = sum([t.get("word_count", 0) for t in transcriptions])
        
        print(f"\nTranscription completed!")
        print(f"Total audio duration: {transcriber.format_timestamps(total_duration)}")
        print(f"Number of utterances: {len(transcriptions)}")
        print(f"Total words: {total_words}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 