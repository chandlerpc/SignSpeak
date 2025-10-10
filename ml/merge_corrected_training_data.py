"""Merge JSON training data with corrections:
1. Use SECOND set of D files (skip first 5 D files from 13:23-13:26)
2. Swap P and Q labels
"""
import json
import os
from datetime import datetime

json_dir = 'C:/FS/JSON'
output_file = './corrected_training_data.json'

# Files to skip (first D recording - timestamp 1759944208 to 1759944364)
SKIP_D_FILES = [
    'asl_training_data_1759944208241.json',  # 13:23:28 - D: 100
    'asl_training_data_1759944220199.json',  # 13:23:40 - D: 200
    'asl_training_data_1759944232068.json',  # 13:23:52 - D: 300
    'asl_training_data_1759944245082.json',  # 13:24:05 - D: 400
    'asl_training_data_1759944256211.json',  # 13:24:16 - D: 500
]

print('='*80)
print('MERGING TRAINING DATA WITH CORRECTIONS')
print('='*80)
print('\nCorrections being applied:')
print('1. Skipping first 5 D files (using only second D recording)')
print('2. Swapping P <-> Q labels (P data becomes Q, Q data becomes P)')
print()

# Initialize merged data
merged_data = {}
stats = {'files_processed': 0, 'files_skipped': 0, 'images_added': 0}

# Get all JSON files
files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

print(f'Processing {len(files)} files...\n')

for filename in files:
    filepath = os.path.join(json_dir, filename)

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Check if this is one of the D files to skip
        skip_d_data = filename in SKIP_D_FILES

        for letter, images in data.items():
            # Skip first D recording
            if letter == 'D' and skip_d_data:
                print(f'SKIPPING: {filename} - D data (first recording)')
                stats['files_skipped'] += 1
                continue

            # Swap P and Q
            corrected_letter = letter
            if letter == 'P':
                corrected_letter = 'Q'
                print(f'SWAPPING: {filename} - P -> Q ({len(images)} images)')
            elif letter == 'Q':
                corrected_letter = 'P'
                print(f'SWAPPING: {filename} - Q -> P ({len(images)} images)')

            # Add to merged data
            if corrected_letter not in merged_data:
                merged_data[corrected_letter] = []

            merged_data[corrected_letter].extend(images)
            stats['images_added'] += len(images)

        stats['files_processed'] += 1

    except Exception as e:
        print(f'ERROR processing {filename}: {e}')

# Final statistics
print('\n' + '='*80)
print('MERGE COMPLETE')
print('='*80)
print(f'\nFiles processed: {stats["files_processed"]}')
print(f'Files skipped: {stats["files_skipped"]}')
print(f'Total images: {stats["images_added"]:,}')
print('\nImages per letter:')

for letter in sorted(merged_data.keys()):
    count = len(merged_data[letter])
    print(f'  {letter}: {count:,} images')

# Save merged data
print(f'\nSaving to {output_file}...')
with open(output_file, 'w') as f:
    json.dump(merged_data, f)

file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f'Saved! File size: {file_size_mb:.1f} MB')
print('\n' + '='*80)
print('READY FOR TRAINING')
print('='*80)
print(f'\nNext step: Run training with:')
print(f'  python retrain_from_json.py {output_file}')