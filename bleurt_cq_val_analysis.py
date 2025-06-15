from filing import Filing
from bleurt import score
import pandas as pd
import os
import re

# Initialize global variables for file paths and model configuration
FILE_DIRECTORY = "generated_cq_files" # for a single file, set to the directory of the file; for a folder with multiple files, set to None
SINGLE_FILE = None # for a single file, set to the path of the file; for a folder with multiple files, set to None
SAVE_DIRECTORY = "bleurt_processed"
MODEL_DIR = "bleurt\BLEURT-20\BLEURT-20"
VALIDATION_FILE = "data/validation.json"
BATCH_SIZE = 1024  # Adjust batch size for scoring


# Extract directory names from the file names for consistent naming
def dir_names_extraction(files_dir):
    dir_names = []
    for file in files_dir:
        base_name = os.path.basename(file)

        match1 = re.search(r'out-([a-z]+)_def_([a-zA-Z]+)-([0-9]+)b', base_name)
        match2 = re.search(r'output_(fewShot|zeroShot)_[a-zA-Z]+_([a-zA-Z]+)-\d+-([0-9]+)b', base_name)
        # for out-*.json
        if match1:
            prefix = match1.group(1) # "out"
            model = match1.group(2) # "gemma"
            size = match1.group(3) # "1"
            dir_names.append(f"{prefix}_{model}_{size}b")
            continue
        # for output_fewShot|zeroShot*
        if match2:
            prefix = match2.group(1) # "fewShot" or "zeroShot"
            model = match2.group(2) # "gemma"
            size = match2.group(3) # "1"
            dir_names.append(f"{prefix}_{model}_{size}b")
            continue
        # If no matches found, use the base name as is
        if match1 is None and match2 is None:
            dir_names.append(base_name)

    return dir_names


# Create a DataFrame from the references, candidates, pair labels, and BLEURT scores
def dataframe(references, candidates, pair_labels, argument_ids, scores):
    return pd.DataFrame({
        "Argument ID": argument_ids,
        "Pair": pair_labels,
        "Reference": references,
        "Candidate": candidates,
        "BLEURT Score": scores
    })


# Create pairs of generated and validation questions, all combinations
def batch_creation(generated_content, validation_content):
    references, candidates, pair_labels, argument_ids = [], [], [], []

    # Iterate through each argument ID in the generated content
    for arg_id, gen_data in generated_content.items():
        gen_cqs = gen_data.get("cqs", [])
        val_data = validation_content.get(arg_id, {})
        val_cqs = val_data.get("cqs", [])

        if not (isinstance(gen_cqs, list) and isinstance(val_cqs, list)): # If there are no cqs, skip
            continue

        # Create pairs of generated and validation questions
        for i, gen_q_obj in enumerate(gen_cqs):
            gen_q = gen_q_obj.get("cq", "").strip()
            if not gen_q:
                continue

            for j, val_q_obj in enumerate(val_cqs):
                val_q = val_q_obj.get("cq", "").strip()
                if not val_q:
                    continue

                references.append(val_q)
                candidates.append(gen_q)
                pair_labels.append(f"gen_q{i}-val_q{j}")
                argument_ids.append(arg_id)

    return references, candidates, pair_labels, argument_ids


# Batched scoring function to handle large datasets efficiently
def batched_score(scorer, references, candidates, batch_size=BATCH_SIZE):
    scores = []
    for i in range(0, len(references), batch_size):
        batch_refs = references[i:i+batch_size]
        batch_cands = candidates[i:i+batch_size]
        batch_scores = scorer.score(references=batch_refs, candidates=batch_cands)
        scores.extend(batch_scores)
    return scores


# Function to process each file, score it with BLEURT, and save results
def process_file(idx, file, filing, scorer):
    print(f"[{idx}] Processing file: {file}")
    try:
        validation_content = filing.open_file(VALIDATION_FILE)
        content = filing.open_file(file)

        references, candidates, pair_labels, argument_ids = batch_creation(content, validation_content)
        print(f"[{idx}] No valid reference-candidate pairs found. Skipping file." if not references else f"[{idx}] Created {len(references)} pairs.")

        scores = batched_score(scorer, references, candidates)

        dir_name = dir_names_extraction([file])[0]
        df = dataframe(references, candidates, pair_labels, argument_ids, scores)
        filing.save_file(dir_name, df)

        print(f"[{idx}] Done processing: {file}")

    except Exception as e:
        print(f"[{idx}] ERROR processing file {file}: {e}")


# Main function to initiate the BLEURT scoring process
def main():
    filing = Filing(FILE_DIRECTORY, SINGLE_FILE, SAVE_DIRECTORY)
    files_dir = filing.file_extraction()
    try:
        scorer = score.BleurtScorer(MODEL_DIR)
    except Exception as e:
        print(f"Error initializing BLEURT scorer: {e}")
        return

    print("No files found to process." if not files_dir else f"Found {len(files_dir)} files to process.")
    print(f"Processing files in directory: {FILE_DIRECTORY}" if FILE_DIRECTORY else f"Processing single file: {SINGLE_FILE}")

    # Process each file in the directory or the single file
    for idx, file in enumerate(files_dir):
        process_file(idx, file, filing, scorer)


if __name__ == "__main__":
    main()
