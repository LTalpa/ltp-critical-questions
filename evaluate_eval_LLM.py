from filing import Filing
import pandas as pd
import os
import re

# Configuration
FILE_DIRECTORY = "output_files/bleurt_files_NO_HALLUCI_BLEURT_scores" # for a single file, set to the directory of the file; for a folder with multiple files, set to None
SINGLE_FILE = None # for a single file, set to the path of the file; for a folder with multiple files, set to None
EVALUATION_DIRECTORY = "evaluated-one-third"
SAVE_DIRECTORY = "eval_analysis"
PREDICTION_COLUMNS = ["tiiuae/Falcon3-1B-Instruct", "prediction"] # Columns to check for predictions in evaluation files


# Identify correct prediction column
def match_prediction_column(columns):
    for col in PREDICTION_COLUMNS:
        if col in columns:
            return col
    raise ValueError(f"None of the expected prediction columns found. Available columns: {columns}")


# Extract model identifier from file name
def extract_model_identifier(file):
    match = re.search(r'([a-zA-Z]+_[a-zA-Z]+_[0-9]+b)', os.path.basename(file))
    return match.group(1) if match else None


# Find evaluation file based on model identifier
def find_eval_file(file, filing):
    if not EVALUATION_DIRECTORY:
        return None
    identifier = extract_model_identifier(file)
    if not identifier:
        print(f"WARNING: Could not extract model identifier from {file}")
        return None

    try:
        for eval_file in filing.file_extraction(EVALUATION_DIRECTORY):
            if identifier.lower() in os.path.basename(eval_file).lower():
                return eval_file
    except Exception as e:
        print(f"ERROR extracting evaluation file for {file}: {e}")
    return None


# Append prediction labels to content DataFrame
def append_predictions(content, eval_df):
    eval_df.columns = eval_df.columns.str.strip()
    prediction_col = match_prediction_column(eval_df.columns)

    eval_df = eval_df.dropna(subset=["intervention_id", "cq_id"]).copy()
    eval_df["intervention_id"] = eval_df["intervention_id"].astype(str).str.strip()
    eval_df["cq_id"] = eval_df["cq_id"].astype(int)

    eval_map = {
        (row["intervention_id"], row["cq_id"]): row[prediction_col]
        for _, row in eval_df.iterrows()
    }

    # Prepare content DataFrame
    content["cq_id"] = content["ID GQ"].astype(str).str.extract(r'gen_q(\d+)').astype(float).astype("Int64")
    content["intervention_id"] = content["Argument ID"].astype(str).str.strip()

    # Map predictions
    content["prediction_label"] = content.apply(
        lambda row: eval_map.get((row["intervention_id"], row["cq_id"])), axis=1
    )

    return content


# Process a single file
def process_file(idx, file, filing):
    print(f"[{idx}] Processing file: {file}")
    try:
        content = filing.open_file(file)
        eval_file = find_eval_file(file, filing)
        if not eval_file:
            print(f"[{idx}] Skipping: No matching evaluation file.")
            return
        eval_df = filing.open_file(eval_file)
        eval_df[match_prediction_column(eval_df.columns)] = eval_df[match_prediction_column(eval_df.columns)].astype(str)

        content = append_predictions(content, eval_df)
        filing.save_file(file, content, index=False, prefix="processed_")
    except Exception as e:
        print(f"[{idx}] ERROR: {e}")


# Main execution
def main():
    filing = Filing(FILE_DIRECTORY, SINGLE_FILE, SAVE_DIRECTORY)
    try:
        files = filing.file_extraction()
    except Exception as e:
        print(f"Error extracting files: {e}")
        return

    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files to process.")
    for idx, file in enumerate(files):
        process_file(idx, file, filing)


if __name__ == "__main__":
    main()
