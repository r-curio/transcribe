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
    
    # Check if they have the same IDs
    text_ids = [seg_id for _, seg_id in text_segments]
    segments_ids = [seg_id for _, seg_id in segments_file]
    
    # Check for exact match
    if text_ids == segments_ids:
        print("\n✅ All segments match perfectly and are in the same order!")
        
        # Check for any gaps in numbering
        print("\nChecking for gaps in segment numbering...")
        all_numbers = []
        for seg_id in text_ids:
            # Extract the number part after the last underscore
            number_part = seg_id.split('_')[-1]
            try:
                all_numbers.append(int(number_part))
            except ValueError:
                print(f"Warning: Could not parse number from {seg_id}")
        
        if all_numbers:
            all_numbers.sort()
            print(f"Segment numbers range from {min(all_numbers)} to {max(all_numbers)}")
            
            # Check for gaps
            expected_numbers = set(range(min(all_numbers), max(all_numbers) + 1))
            actual_numbers = set(all_numbers)
            missing_numbers = expected_numbers - actual_numbers
            
            if missing_numbers:
                print(f"Missing segment numbers: {sorted(missing_numbers)}")
            else:
                print("No gaps in segment numbering found")
                
    else:
        print("\n❌ There are mismatches between the files")
        
        # Find first mismatch
        for i, (text_id, seg_id) in enumerate(zip(text_ids, segments_ids)):
            if text_id != seg_id:
                print(f"First mismatch at position {i+1}:")
                print(f"  Text file: {text_id}")
                print(f"  Segments file: {seg_id}")
                break

if __name__ == "__main__":
    main() 