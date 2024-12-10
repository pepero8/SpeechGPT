import sys
sys.path.append("/home/jhwan98/EmoSDS/SpeechGPT/speechgpt")
from utils.speech2unit.speech2unit import Speech2Unit
import os
import random
from pathlib import Path
import argparse


def split_file(input_file, output_path, train_ratio=0.8, shuffle=True, seed=9814):
    # Set random seed for reproducibility
    random.seed(seed)

    # Read all lines from the input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Shuffle lines if requested
    if shuffle:
        random.shuffle(lines)

    # Calculate split index
    split_idx = int(len(lines) * train_ratio)

    # Split the lines
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    # Create output file paths
    output_path = Path(output_path)
    train_file = output_path / "train.txt"
    test_file = output_path / "dev.txt"

    # Write training file
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    # Write testing file
    with open(test_file, "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    print("Split completed")

    return train_file, test_file


def process_wav_files(dir_path, out_file, s2u):
	root_dir = Path(dir_path)
    
	count = 0
	with open(out_file, 'a') as f:
		for root, dirs, files in os.walk(root_dir):
			for file in files:
				if file.lower().endswith('.wav'):
					try:
						cur_path = Path(root) / file
						units = s2u(cur_path)
						f.write(f"{units}\n")
						print("processed:", cur_path)
						count += 1
						# if count == 5:
						# 	return
					except Exception as e:
						print(f"Error processing {cur_path}: \n", e)
						return

		print(f"Preprocessing finished. Total data count: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    dir_path = "./styletalk"
    data_file = "./data.txt"
    out_path = "./stage1"

    train_ratio = 0.2
    shuffle = True
    seed = 9814

    s2u = Speech2Unit(
			ckpt_dir=ckpt_dir
		)

    process_wav_files(dir_path, data_file, s2u)

    train_file, test_file = split_file(
        data_file,
		out_path,
        train_ratio=train_ratio,
        shuffle=shuffle,
        seed=seed,
    )
