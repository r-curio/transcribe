#!/usr/bin/env python3

def extract_segment_ids(filename):
    """Extract segment IDs from a file."""
    segment_ids = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract the segment ID (first part before space)
                parts = line.split()
                if parts:
                    segment_ids.add(parts[0])
    return segment_ids

def main():
    # Extract segment IDs from both files
    text_segments = extract_segment_ids('text')
    segments_file = extract_segment_ids('segments')
    
    print(f"Total segments in text file: {len(text_segments)}")
    print(f"Total segments in segments file: {len(segments_file)}")
    
    # Find missing segments
    missing_in_segments = text_segments - segments_file
    missing_in_text = segments_file - text_segments
    
    print(f"\nSegments in text but missing in segments: {len(missing_in_segments)}")
    if missing_in_segments:
        print("Missing segments:")
        for seg in sorted(missing_in_segments):
            print(f"  {seg}")
    
    print(f"\nSegments in segments but missing in text: {len(missing_in_text)}")
    if missing_in_text:
        print("Missing segments:")
        for seg in sorted(missing_in_text):
            print(f"  {seg}")
    
    # Check for exact matches
    if text_segments == segments_file:
        print("\n✅ All segments match perfectly!")
    else:
        print("\n❌ There are mismatches between the files")
        
        # Show some examples of what's in text but not in segments
        if missing_in_segments:
            print(f"\nExamples of text segments not in segments file:")
            for seg in sorted(list(missing_in_segments))[:5]:
                print(f"  {seg}")
        
        # Show some examples of what's in segments but not in text
        if missing_in_text:
            print(f"\nExamples of segments file entries not in text file:")
            for seg in sorted(list(missing_in_text))[:5]:
                print(f"  {seg}")

if __name__ == "__main__":
    main() 