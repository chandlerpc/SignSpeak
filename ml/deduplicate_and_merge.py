"""Deduplicate overlapping JSON files and merge with corrections
Each auto-download contains ALL previous images, so we need to take only the LAST/largest file per letter
"""
import json
import os
from datetime import datetime

json_dir = 'C:/FS/JSON'
output_file = './training_data_deduplicated_500.json'
TARGET_IMAGES_PER_LETTER = 500

# Files to skip (first D recording)
SKIP_D_FILES = [
    'asl_training_data_1759944208241.json',
    'asl_training_data_1759944220199.json',
    'asl_training_data_1759944232068.json',
    'asl_training_data_1759944245082.json',
    'asl_training_data_1759944256211.json',
]

print('='*80)
print('DEDUPLICATING AND MERGING JSON FILES')
print('='*80)
print(f'Target: {TARGET_IMAGES_PER_LETTER} images per letter\n')

# Load all files and organize by letter
files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
print(f'Found {len(files)} JSON files\n')

# Group files by letter and find the largest (most complete) file for each
letter_files = {}

for filename in files:
    filepath = os.path.join(json_dir, filename)

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        for letter, images in data.items():
            if letter not in letter_files:
                letter_files[letter] = []

            letter_files[letter].append({
                'filename': filename,
                'count': len(images),
                'images': images
            })
    except Exception as e:
        print(f'ERROR reading {filename}: {e}')

# For each letter, select the file with exactly 500 images, or the closest to 500
print('Selecting best file for each letter:')
print('='*80)

merged_data = {}

for letter in sorted(letter_files.keys()):
    files_for_letter = letter_files[letter]

    # Skip first D recording
    if letter == 'D':
        # Filter out the skip files
        files_for_letter = [
            f for f in files_for_letter
            if f['filename'] not in SKIP_D_FILES
        ]
        print(f'\nD: Skipped first 5 files (first recording)')

    # Find file with exactly 500 images, or closest to 500
    files_for_letter.sort(key=lambda x: (abs(x['count'] - TARGET_IMAGES_PER_LETTER), -x['count']))
    best_file = files_for_letter[0]

    # Take exactly 500 images (or less if not enough)
    selected_images = best_file['images'][:TARGET_IMAGES_PER_LETTER]

    # Apply P/Q swap
    corrected_letter = letter
    if letter == 'P':
        corrected_letter = 'Q'
        print(f'{letter} -> {corrected_letter}: {len(selected_images)} images from {best_file["filename"]} [SWAPPED]')
    elif letter == 'Q':
        corrected_letter = 'P'
        print(f'{letter} -> {corrected_letter}: {len(selected_images)} images from {best_file["filename"]} [SWAPPED]')
    else:
        print(f'{letter}: {len(selected_images)} images from {best_file["filename"]}')

    merged_data[corrected_letter] = selected_images

# Save merged data FIRST (before any unicode print issues)
print(f'\nSaving to {output_file}...')
with open(output_file, 'w') as f:
    json.dump(merged_data, f)

file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f'Saved! File size: {file_size_mb:.1f} MB')

print('\n' + '='*80)
print('READY FOR TRAINING')
print('='*80)
print(f'\nNext: python retrain_from_json_efficient.py {output_file}')