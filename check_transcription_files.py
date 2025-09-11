#!/usr/bin/env python3
"""
Comprehensive transcription file checker
Checks if all entries in text file have corresponding entries in segments file
and provides detailed analysis of the files.
"""

import sys
import os

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

def check_file_exists(filename):
    """Check if file exists and is readable."""
    if not os.path.exists(filename):
        print(f"❌ Error: File '{filename}' not found")
        return False
    if not os.access(filename, os.R_OK):
        print(f"❌ Error: File '{filename}' is not readable")
        return False
    return True

def analyze_segment_numbers(segment_ids):
    """Analyze segment numbering for gaps and patterns."""
    print("\n📊 Segment Numbering Analysis:")
    
    all_numbers = []
    for seg_id in segment_ids:
        # Extract the number part after the last underscore
        number_part = seg_id.split('_')[-1]
        try:
            all_numbers.append(int(number_part))
        except ValueError:
            print(f"⚠️  Warning: Could not parse number from {seg_id}")
    
    if all_numbers:
        all_numbers.sort()
        print(f"  • Segment numbers range from {min(all_numbers)} to {max(all_numbers)}")
        print(f"  • Total unique segment numbers: {len(set(all_numbers))}")
        
        # Check for gaps
        expected_numbers = set(range(min(all_numbers), max(all_numbers) + 1))
        actual_numbers = set(all_numbers)
        missing_numbers = expected_numbers - actual_numbers
        
        if missing_numbers:
            print(f"  • Missing segment numbers: {sorted(missing_numbers)}")
            print(f"  • Gap count: {len(missing_numbers)}")
        else:
            print("  • No gaps in segment numbering found")
            
        # Check for duplicates
        from collections import Counter
        duplicates = [num for num, count in Counter(all_numbers).items() if count > 1]
        if duplicates:
            print(f"  • Duplicate segment numbers: {sorted(duplicates)}")
        else:
            print("  • No duplicate segment numbers found")

def check_transcription_files(text_file='text', segments_file='segments'):
    """Main function to check transcription files."""
    print("🔍 Transcription Files Checker")
    print("=" * 50)
    
    # Check if files exist
    if not check_file_exists(text_file):
        return False
    if not check_file_exists(segments_file):
        return False
    
    # Extract segment IDs from both files
    print(f"\n📁 Reading files...")
    text_segments = extract_segment_ids(text_file)
    segments_file_data = extract_segment_ids(segments_file)
    
    print(f"  • {text_file}: {len(text_segments)} segments")
    print(f"  • {segments_file}: {len(segments_file_data)} segments")
    
    # Extract just the IDs for comparison
    text_ids = [seg_id for _, seg_id in text_segments]
    segments_ids = [seg_id for _, seg_id in segments_file_data]
    
    # Check for exact match
    if text_ids == segments_ids:
        print("\n✅ All segments match perfectly and are in the same order!")
        
        # Analyze segment numbering
        analyze_segment_numbers(text_ids)
        
        # Show sample entries
        print(f"\n📋 Sample entries from {text_file}:")
        for i in range(min(5, len(text_segments))):
            line_num, seg_id = text_segments[i]
            print(f"  Line {line_num}: {seg_id}")
            
        print(f"\n📋 Sample entries from {segments_file}:")
        for i in range(min(5, len(segments_file_data))):
            line_num, seg_id = segments_file_data[i]
            print(f"  Line {line_num}: {seg_id}")
            
        return True
        
    else:
        print("\n❌ There are mismatches between the files")
        
        # Find first mismatch
        for i, (text_id, seg_id) in enumerate(zip(text_ids, segments_ids)):
            if text_id != seg_id:
                print(f"First mismatch at position {i+1}:")
                print(f"  {text_file}: {text_id}")
                print(f"  {segments_file}: {seg_id}")
                break
        
        # Find missing segments
        text_ids_set = set(text_ids)
        segments_ids_set = set(segments_ids)
        
        missing_in_segments = text_ids_set - segments_ids_set
        missing_in_text = segments_ids_set - text_ids_set
        
        print(f"\n🔍 Missing segments analysis:")
        print(f"  • Segments in {text_file} but missing in {segments_file}: {len(missing_in_segments)}")
        if missing_in_segments:
            print("    Missing segments:")
            for seg in sorted(missing_in_segments):
                # Find the line number in text file
                line_num = next(line_num for line_num, seg_id in text_segments if seg_id == seg)
                print(f"      {seg} (line {line_num} in {text_file})")
        
        print(f"  • Segments in {segments_file} but missing in {text_file}: {len(missing_in_text)}")
        if missing_in_text:
            print("    Missing segments:")
            for seg in sorted(missing_in_text):
                # Find the line number in segments file
                line_num = next(line_num for line_num, seg_id in segments_file_data if seg_id == seg)
                print(f"      {seg} (line {line_num} in {segments_file})")
        
        return False

def main():
    """Main entry point."""
    if len(sys.argv) == 1:
        # Use default filenames
        success = check_transcription_files()
    elif len(sys.argv) == 3:
        # Use provided filenames
        text_file = sys.argv[1]
        segments_file = sys.argv[2]
        success = check_transcription_files(text_file, segments_file)
    else:
        print("Usage: python3 check_transcription_files.py [text_file] [segments_file]")
        print("If no arguments provided, uses 'text' and 'segments' as default filenames")
        return
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 