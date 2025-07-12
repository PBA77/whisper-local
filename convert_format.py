#!/usr/bin/env python3
import json
import sys
from transcribe_diarize import OutputFormatter

if len(sys.argv) != 4:
    print("Usage: python convert_format.py input.json output_format output_file")
    sys.exit(1)

input_file = sys.argv[1]
output_format = sys.argv[2]
output_file = sys.argv[3]

with open(input_file, 'r') as f:
    results = json.load(f)

if output_format == "srt":
    content = OutputFormatter.to_srt(results)
elif output_format == "csv":
    content = OutputFormatter.to_csv(results)
else:
    content = OutputFormatter.to_json(results)

with open(output_file, 'w') as f:
    f.write(content)

print(f"Converted to {output_format} format: {output_file}")