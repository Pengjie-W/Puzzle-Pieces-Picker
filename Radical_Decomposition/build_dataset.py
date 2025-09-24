import os
import json
import copy
import argparse


def build_dataset(input_dir: str, output_path: str) -> None:
    """
    Walk the entire ``input_dir`` and build a dataset manifest.

    Logic mirrors the original script:
    - use the second-to-last folder name in the path as the label;
    - if a directory has exactly two files and one is ``0.jpg``, add only the
      current file once and stop iterating that directory;
    - otherwise write every file in the directory to the manifest.
    """
    dataset = []

    for root, _, files in os.walk(input_dir):
        # Sort files for deterministic behavior
        files_sorted = sorted(files)
        for fname in files_sorted:
            file_path = os.path.join(root, fname)
            if not os.path.isfile(file_path):
                continue

            second_to_last_folder = os.path.basename(os.path.dirname(file_path))
            record = {
                "path": file_path,
                "label": second_to_last_folder,
            }

            # Reproduce the original condition for the special two-file case
            if ('0.jpg' in files) and (len(files) == 2):
                dataset.append(copy.deepcopy(record))
                break

            dataset.append(copy.deepcopy(record))

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(len(dataset), "items found.")
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Build the Decomposition dataset manifest')
    parser.add_argument('--input_dir', default='./output/Decomposition', help='Root directory to scan')
    parser.add_argument('--output', default='./output/Decomposition_Dataset.json', help='Output JSON file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_dataset(args.input_dir, args.output)
    print(f"[OK] Generated: {args.output}")
