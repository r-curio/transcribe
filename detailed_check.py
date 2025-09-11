#!/usr/bin/env python3

def extract_segment_ids(filename):
    """Extract segment IDs from a file."""
    segment_ids = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                # Extract the segment ID (first part before space)
                parts = line.split()
                if parts:
                    segment_ids.append((line_num, parts[0]))
    return segment_ids

def main():
    # Extract segment IDs from both files
    text_segments = extract_segment_ids('text')
    segments_file = extract_segment_ids('segments')
    
    print(f"Total segments in text file: {len(text_segments)}")
    print(f"Total segments in segments file: {len(segments_file)}")
    
    # Extract just the IDs for comparison
    text_ids = {seg_id for _, seg_id in text_segments}
    segments_ids = {seg_id for _, seg_id in segments_file}
    
    # Find missing segments
    missing_in_segments = text_ids - segments_ids
    missing_in_text = segments_ids - text_ids
    
    print(f"\nSegments in text but missing in segments: {len(missing_in_segments)}")
    if missing_in_segments:
        print("Missing segments:")
        for seg in sorted(missing_in_segments):
            # Find the line number in text file
            line_num = next(line_num for line_num, seg_id in text_segments if seg_id == seg)
            print(f"  {seg} (line {line_num} in text)")
    
    print(f"\nSegments in segments but missing in text: {len(missing_in_text)}")
    if missing_in_text:
        print("Missing segments:")
        for seg in sorted(missing_in_text):
            # Find the line number in segments file
            line_num = next(line_num for line_num, seg_id in segments_file if seg_id == seg)
            print(f"  {seg} (line {line_num} in segments)")
    
    # Check for exact matches
    if text_ids == segments_ids:
        print("\n✅ All segments match perfectly!")
        
        # Show some sample entries from both files
        print(f"\nSample entries from text file:")
        for i in range(min(5, len(text_segments))):
            line_num, seg_id = text_segments[i]
            print(f"  Line {line_num}: {seg_id}")
            
        print(f"\nSample entries from segments file:")
        for i in range(min(5, len(segments_file))):
            line_num, seg_id = segments_file[i]
            print(f"  Line {line_num}: {seg_id}")
    else:
        print("\n❌ There are mismatches between the files")

if __name__ == "__main__":
    main() 