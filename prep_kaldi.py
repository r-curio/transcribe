import re

input_file = "84_xx10xxxx_15A_transcription.txt"
recording_id = "84_xx10xxxx_15A"   # same as wav file name without extension

segments_out = open("segments", "w", encoding="utf-8")
text_out = open("text", "w", encoding="utf-8")

utterance_count = 0
current_utterance = {}

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        
        # Skip empty lines and header
        if not line or line.startswith("Transcription of") or line.startswith("Model:") or line.startswith("Total utterances:") or line.startswith("="):
            continue
        
        # Check for utterance start
        utterance_match = re.match(r"Utterance (\d+):", line)
        if utterance_match:
            # If we have a previous complete utterance, write it out
            if current_utterance and 'start' in current_utterance and 'end' in current_utterance and 'text' in current_utterance:
                utterance_count += 1
                utt_id = f"{recording_id}_{utterance_count:04d}"
                
                # Write to segments file: utt_id rec_id start end
                segments_out.write(f"{utt_id} {recording_id} {current_utterance['start']:.3f} {current_utterance['end']:.3f}\n")
                
                # Write to text file: utt_id transcript
                text_out.write(f"{utt_id} {current_utterance['text']}\n")
            
            # Start new utterance
            current_utterance = {}
            continue
        
        # Parse time information
        time_match = re.match(r"Time: \[(\d+):(\d+\.\d+) - (\d+):(\d+\.\d+)\]", line)
        if time_match:
            start_min, start_sec, end_min, end_sec = time_match.groups()
            current_utterance['start'] = int(start_min) * 60 + float(start_sec)
            current_utterance['end'] = int(end_min) * 60 + float(end_sec)
            continue
        
        # Parse text
        text_match = re.match(r"Text: (.+)", line)
        if text_match:
            current_utterance['text'] = text_match.group(1).strip()
            continue

# Handle the last utterance
if current_utterance and 'start' in current_utterance and 'end' in current_utterance and 'text' in current_utterance:
    utterance_count += 1
    utt_id = f"{recording_id}_{utterance_count:04d}"
    
    # Write to segments file: utt_id rec_id start end
    segments_out.write(f"{utt_id} {recording_id} {current_utterance['start']:.3f} {current_utterance['end']:.3f}\n")
    
    # Write to text file: utt_id transcript
    text_out.write(f"{utt_id} {current_utterance['text']}\n")

segments_out.close()
text_out.close()

print(f"Processed {utterance_count} utterances")
print(f"Created segments and text files for recording: {recording_id}")
