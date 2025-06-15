"""
Analyse per model:
    - Which prompt type was best (check based on labels, can compare with hallucination)
    - Which prompt hallucinated most

Analyse between models:
    - Which model performed best overall, neglecting prompts (based on labels, can compare with hallucinations)
    - Which prompt resulted in the most hallucinations  
"""
from filing import Filing
import pandas as pd


FILE_DIRECTORY = "eval_analysis" # for a single file, set to the directory of the file; for a folder with multiple files, set to None
SINGLE_FILE = None # for a single file, set to the path of the file; for a folder with multiple files, set to None
COLUMN = "prediction_label" # "Label" for "Useful", "Unhelpful", or "Invalid" or "hallucinated" for "yes" (Hallucinated) or "no"(Not Hallucinated)
SAVE_DIRECTORY = f"MODEL_{COLUMN}" if "model" in FILE_DIRECTORY else f"PROMPT_{COLUMN}"

# Generate an average prompt descriptions per label per model in a file
def average_data(filing,all_dfs_pm, model_dir):
    concatenated = pd.concat(all_dfs_pm)
    averaged = concatenated.groupby(concatenated.index).mean()
    file_name = f"{model_dir}_average_describe.xlsx"
    filing.save_file(file_name, averaged, index=True)


# This function summarises on prompt type per model
def analyse_per_label(filing, file):
    content = filing.open_file(file)
    # calculate the mean confidence score per file
    labels = content.groupby(f"{COLUMN}").describe() # 
    filing.save_file(file, labels, index=True)
    return labels


# Main function to process all models and their prompts
def main():
    filing = Filing(FILE_DIRECTORY, SINGLE_FILE, SAVE_DIRECTORY)

    # Go over each subdirectory in the main directory
    model_dirs = filing.file_extraction()
    for model_dir in model_dirs:
        all_dfs_pm = []
        # Go over each file in the directory
        files = filing.file_extraction(model_dir)
        for file in files:
            all_dfs_pm.append(analyse_per_label(filing, file))
        average_data(filing, all_dfs_pm, model_dir)
    print("All models processed successfully.") if "model" in FILE_DIRECTORY  else print("All prompts processed successfully.")


if __name__ == "__main__":
    main()