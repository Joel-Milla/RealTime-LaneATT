import json


def convert_lane_format(data):
    """Convert lane format from json1 style to json2 style"""
    # Original h_samples and lanes
    old_h = data['h_samples']
    old_lanes = data['lanes']

    # Create new h_samples from 0 to 710 with step 10
    new_h = list(range(0, 720, 10))

    # Calculate padding needed at the beginning
    start_idx = new_h.index(old_h[0])

    # Convert each lane
    new_lanes = []
    for lane in old_lanes:
        # Pad with -2 at the beginning, keep original values
        new_lane = [-2] * start_idx + lane # concatenates an array of -2 to the lane of file
        # Extend to match new h_samples length if needed, -2 at end if needed
        while len(new_lane) < len(new_h):
            new_lane.append(-2)
        new_lanes.append(new_lane)

    return {
        'lanes': new_lanes,
        'h_samples': new_h,
        'raw_file': data['raw_file']
    }


def process_jsonl_file(input_file, output_file):
    """Process JSONL file (one JSON per line)"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Parse each line as JSON
            data = json.loads(line.strip())
            # Convert format
            converted = convert_lane_format(data)
            # Write as single line JSON
            outfile.write(json.dumps(converted) + '\n')


def verify_conversion(original_file, output_file):
    """Verify that all non-(-2) points match between original and output"""
    print("Verifying conversion...")

    with open(original_file, 'r') as orig, open(output_file, 'r') as out:
        line_num = 0
        for orig_line, out_line in zip(orig, out):
            line_num += 1
            orig_data = json.loads(orig_line.strip())
            out_data = json.loads(out_line.strip())

            # Get original points
            orig_h = orig_data['h_samples']
            orig_lanes = orig_data['lanes']

            # Get output points
            out_h = out_data['h_samples']
            out_lanes = out_data['lanes']

            # Check each lane
            for lane_idx, (orig_lane, out_lane) in enumerate(zip(orig_lanes, out_lanes)):
                # Extract non-(-2) points from original
                orig_points = [(orig_lane[i], orig_h[i]) for i in range(len(orig_lane)) if orig_lane[i] != -2]

                # Extract non-(-2) points from output
                out_points = [(out_lane[i], out_h[i]) for i in range(len(out_lane)) if out_lane[i] != -2]

                # Compare
                if orig_points != out_points:
                    print(f"DISCREPANCY in line {line_num}, lane {lane_idx}:")
                    print(f"  File: {orig_data['raw_file']}")
                    print(f"  Original points: {orig_points[:3]}... (showing first 3)")
                    print(f"  Output points: {out_points[:3]}... (showing first 3)")
                    return False

    print(f"✓ Verification complete! All {line_num} lines match perfectly.")
    return True

# Example usage
if __name__ == "__main__":
    # For JSONL files (one JSON per line)
    # process_jsonl_file('labels.json', 'output.json')

    # Verify the conversion
    verify_conversion('../dataset_generator/train/labels.json', 'output.json')
