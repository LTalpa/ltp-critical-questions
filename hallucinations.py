from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from filing import Filing
from tqdm import tqdm
import pandas as pd
import torch
import os
import re


FILE_DIRECTORY = "generated_cq_files"  # Folder with files, or None
SINGLE_FILE = None                    # Path to a single file, or None
SAVE_DIRECTORY = "hallucination_reports"  # Directory to save the results
MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"


# Load the hallucination classification model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


# Hallucination classifier
def classify_hallucination(intervention, cq, tokenizer, model, device):
    inputs = tokenizer(intervention, cq, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_label].item()
        return predicted_label, confidence


# Process a single file and return a DataFrame of results
def process_file(file, dir_name, tokenizer, model, device, filing, idx):
    data = filing.open_file(file)
    records = []

    for key, entry in tqdm(data.items(), desc=f"[{idx}] Processing entries for {dir_name}"):
        intervention = entry.get("intervention", "")
        cqs = entry.get("cqs", [])

        if not isinstance(cqs, list):
            continue

        for cq_obj in cqs:
            if isinstance(cq_obj, dict) and "cq" in cq_obj:
                cq_text = cq_obj["cq"]
                cq_id = cq_obj.get("id", "")

                label, confidence = classify_hallucination(intervention, cq_text, tokenizer, model, device)

                records.append({
                    "intervention_id": key,
                    "intervention": intervention,
                    "cq_id": cq_id,
                    "cq": cq_text,
                    "hallucinated": "Yes" if label == 1 else "No",
                    "confidence_score": confidence
                })

    df = pd.DataFrame(records)
    return df


# Extract directory names from file names for consistent naming
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


# Main function to manage the hallucination classification workflow
def main():
    filing = Filing(FILE_DIRECTORY, SINGLE_FILE, SAVE_DIRECTORY)
    files_dir = filing.file_extraction()
    
    dir_names = dir_names_extraction(files_dir)

    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

    print("No files found to process." if not files_dir else f"Found {len(files_dir)} files to process.")
    print(f"Processing files in directory: {FILE_DIRECTORY}" if FILE_DIRECTORY else f"Processing single file: {SINGLE_FILE}")

    for idx, file in enumerate(files_dir):
        try:
            df = process_file(file, dir_names[idx], tokenizer, model, device, filing, idx)
            filing.save_file(dir_names[idx] + "_hallucination_report_with_confidence", df)
            print(f"[{idx}] Done processing: {dir_names[idx]}")
        except Exception as e:
            print(f"[{idx}] ERROR processing file {file}: {e}")


if __name__ == "__main__":
    main()
