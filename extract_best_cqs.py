from filing import Filing
import os
import re


FILE_DIRECTORY = "all_outputs" # for a single file, set to the directory of the file; for a folder with multiple files, set to None
SINGLE_FILE = None # for a single file, set to the path of the file; for a folder with multiple files, set to None
HALLUCINATION_THRESHOLD = 0.9 # Set to None if you do not want hallucinations removed, set to value between 0 (no hallucination) to 1 (hallucination) to remove CQs above threshold
HALLUCINATION_DIRECTORY = "hallucination_reports" # If threshold not set to None, set the directory of the hallucination files
VALIDATION_FILE = "data/validation.json"
SAVE_DIRECTORY = "best_cqs_extracted"


def find_label(argument_id, reference, validation):
    if argument_id not in validation:
        return None
    for cq_entry in validation[argument_id].get("cqs", []):
        if cq_entry["cq"].strip() == reference.strip():
            return cq_entry["label"]
    return None

# Adds label "Useful", "Unhelpful", or "Invalid" to all pair options in the Excel file
def add_labels(content, validation):
    content["Label"] = content.apply(lambda row: find_label(row["Argument ID"], row["Reference"], validation), axis=1)
    return content


# Extracts the maximum BLEURT score for each generated question from the content
def extract_max(content):
    # Extract 'gen_qX' from 'Pair' to extract the highest BLEURT score for each generated question
    content["Gen Question"] = content["Pair"].apply(
        lambda x: re.match(r'(gen_q\d+)', x).group(1) if re.match(r'(gen_q\d+)', x) else None
    )
    content = content.dropna(subset=["Gen Question"])

    # Get row with max BLEURT score per (Argument ID, Gen Question)
    idx_max_bleurt = content.groupby(["Argument ID", "Gen Question"])["BLEURT Score"].idxmax()
    best_rows = content.loc[idx_max_bleurt, [
        "Argument ID", "Gen Question", "Reference", "Candidate", "BLEURT Score", "Label"
    ]].copy()

    # Rename for clarity
    best_rows = best_rows.rename(columns={
        "Gen Question": "ID GQ",
        "Reference": "Generated Question",
        "Candidate": "Validation Question",
        "BLEURT Score": "Max BLEURT Score"
    })
    return best_rows


# Extracts directory names from the file names for consistent naming
def dir_names_extraction(files_dir):
    dir_names = []
    for file in files_dir:
        base_name = os.path.basename(file)
        match1 = re.search(r'out-([a-z]+)_def_([a-zA-Z]+)-([0-9]+)b', base_name)
        match2 = re.search(r'output_(fewShot|zeroShot)_[a-zA-Z]+_([a-zA-Z]+)-\d+-([0-9]+)b', base_name)
        if match1:
            dir_names.append(f"{match1.group(1)}_{match1.group(2)}_{match1.group(3)}b")
        elif match2:
            dir_names.append(f"{match2.group(1)}_{match2.group(2)}_{match2.group(3)}b")
        else:
            dir_names.append(os.path.splitext(base_name)[0])  # fallback to base name
    return dir_names


# Matches the file name to extract the model identifier (e.g., fewShot_llama_1b, acot_gemma_4b) used for hallucination file matching
def match_files(file):
    # Match something like fewShot_llama_1b, acot_gemma_4b, etc.
    base_file = os.path.basename(file)
    match = re.search(r'([a-zA-Z]+_[a-zA-Z]+_[0-9]+b)', base_file)
    return match.group(1) if match else None


# Remove the hallucinated questions above the specified threshold from the data before extracting best CQs
def remove_hallucinations(content, file, filing, threshold):
    hallucination_df = filing.open_file(file)
    print("Hallucination file columns:", hallucination_df.columns.tolist())
    hallucinated = hallucination_df[
        (hallucination_df['confidence_score'] >= threshold)
    ]
    required_columns = {'intervention_id', 'cq_id', 'confidence_score'}
    if not required_columns.issubset(hallucination_df.columns):
        raise ValueError(f"Missing columns in hallucination file: expected {required_columns}, got {hallucination_df.columns.tolist()}")


    if hallucinated.empty:
        print(f"WARNING: No hallucinations found in {os.path.basename(file)} above threshold {threshold}.")
        return content

    # Ensure matching on proper string columns
    hallucinated_pairs = set(zip(
        hallucinated['intervention_id'],
        hallucinated['cq_id'].astype(str).apply(lambda x: f"gen_q{x}")
    ))
    print(f"Total hallucinated pairs found: {len(hallucinated_pairs)}")
    before = len(content)
    content_filtered = content[~content.apply(lambda row: (row["Argument ID"], row["Gen Question"]) in hallucinated_pairs, axis=1)]
    after = len(content_filtered)
    print(f"Filtered {before - after} rows from {before} based on hallucination pairs.")


    # Keep rows that do NOT match hallucinated pairs
    content_filtered = content[
        ~content.apply(lambda row: (row["Argument ID"], row["Gen Question"]) in hallucinated_pairs, axis=1)
        ]

    return content_filtered


# Remove hallucinations from the file if a threshold AND valid directory is set
def find_hallu_file(content, file, filing, threshold):
    if HALLUCINATION_DIRECTORY is None or threshold is None:
        return content  # Nothing to filter

    content["Gen Question"] = content["Pair"].apply(
        lambda x: re.match(r'(gen_q\d+)', x).group(1) if re.match(r'(gen_q\d+)', x) else None
    )
    content = content.dropna(subset=["Gen Question"])

    identifier = match_files(file)
    if not identifier:
        print(f"WARNING: Could not extract model identifier from {file}")
        return content

    try:
        hallucination_files = filing.file_extraction(HALLUCINATION_DIRECTORY)
        matched_file = None
        for hallu_file in hallucination_files:
            hallu_basename = os.path.basename(hallu_file)
            if identifier.lower() in hallu_basename.lower():
                matched_file = hallu_file
                print(f"Matched hallucination file: {matched_file}")
                break
        else:
            print(f"WARNING: No matching hallucination file found for identifier: {identifier}")


        if not matched_file:
            print(f"WARNING: No matching hallucination file found for {file}")
            return content

        content_filtered = remove_hallucinations(content, matched_file, filing, threshold)
        print(f"Filtered hallucinated questions for {file}. Remaining rows: {len(content_filtered)}")
        return content_filtered

    except Exception as e:
        print(f"ERROR during hallucination file extraction for {file}: {e}")
        return content


# Process each file to extract the best CQs based on BLEURT scores and optionally filter hallucinations
def process_file(idx, file, filing):
    print(f"[{idx}] Processing file: {file}")
    try:
        content = filing.open_file(file)
        validation = filing.open_file(VALIDATION_FILE)
    except Exception as e:
        print(f"[{idx}] ERROR opening file: {e}")
        return

    try:
        content["BLEURT Score"] = content["BLEURT Score"].astype(float)

        no_hallucinations = False
        if HALLUCINATION_THRESHOLD is not None:
            content = find_hallu_file(content, file, filing, threshold=HALLUCINATION_THRESHOLD)
            no_hallucinations = True

        content = add_labels(content, validation)
        best_rows = extract_max(content)
        if not best_rows.empty:
            prefix = "no_hallu_" if no_hallucinations else ""
            filing.save_file(file, best_rows, prefix=prefix)
        else:
            print(f"[{idx}] No valid rows found in {file} after processing.")

    except Exception as e:
        print(f"[{idx}] ERROR during processing: {e}")


# main function to execute the script
def main():
    filing = Filing(FILE_DIRECTORY, SINGLE_FILE, SAVE_DIRECTORY)
    try:
        files = filing.file_extraction()
    except Exception as e:
        print(f"Error extracting files: {e}")
        return

    if not files:
        print("No files found to process.")
        return

    print(f"Found {len(files)} files to process.")
    for idx, file in enumerate(files):
        process_file(idx, file, filing)
        break


if __name__ == "__main__":
    main()
