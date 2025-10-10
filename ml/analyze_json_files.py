"""Analyze JSON files to identify letters and prepare for merging"""
import json
import os
from datetime import datetime

json_dir = 'C:/FS/JSON'
files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

print(f'Analyzing {len(files)} JSON files...\n')

# Quick analysis - just read keys, not full data
file_info = []
for filename in files:
    filepath = os.path.join(json_dir, filename)
    try:
        with open(filepath, 'r') as f:
            # Read just the beginning to get keys
            data = json.load(f)

        letters = list(data.keys())
        img_counts = {k: len(v) for k, v in data.items()}

        # Extract timestamp
        timestamp = int(filename.split('_')[-1].replace('.json', ''))
        time = datetime.fromtimestamp(timestamp/1000).strftime('%H:%M:%S')

        file_info.append({
            'filename': filename,
            'letters': letters,
            'counts': img_counts,
            'time': time,
            'timestamp': timestamp
        })

    except Exception as e:
        print(f'ERROR reading {filename}: {e}')

# Group by letter
by_letter = {}
for info in file_info:
    for letter in info['letters']:
        if letter not in by_letter:
            by_letter[letter] = []
        by_letter[letter].append(info)

print('='*80)
print('LETTER SUMMARY:')
print('='*80)
for letter in sorted(by_letter.keys()):
    files_for_letter = by_letter[letter]
    total_images = sum(f['counts'][letter] for f in files_for_letter)
    print(f'\n{letter}: {len(files_for_letter)} file(s), {total_images} total images')
    for f in files_for_letter:
        print(f'  {f["time"]} - {f["filename"][-30:]:35s} {f["counts"][letter]:4d} images')

# Identify issues
print('\n' + '='*80)
print('ISSUES TO ADDRESS:')
print('='*80)

# Find D files
if 'D' in by_letter and len(by_letter['D']) > 1:
    print(f'\n✓ Found {len(by_letter["D"])} D files (will use the SECOND one):')
    for i, f in enumerate(by_letter['D'], 1):
        marker = ' <-- USE THIS ONE' if i == 2 else ' <-- SKIP'
        print(f'  {i}. {f["filename"]} ({f["counts"]["D"]} images){marker}')

# Find P and Q files
if 'P' in by_letter:
    print(f'\n✓ Found {len(by_letter["P"])} P file(s) (actually Q):')
    for f in by_letter['P']:
        print(f'  {f["filename"]} ({f["counts"]["P"]} images)')

if 'Q' in by_letter:
    print(f'\n✓ Found {len(by_letter["Q"])} Q file(s) (actually P):')
    for f in by_letter['Q']:
        print(f'  {f["filename"]} ({f["counts"]["Q"]} images)')

print('\n' + '='*80)
print('READY TO MERGE')
print('='*80)