"""
Analyse per model:
    - Which prompt type was best
Analyse between models:
    - Which model performed best overall
"""
from filing import Filing
import pandas as pd


FILE_DIRECTORY = "eval_analysis_prompt" # for a single file, set to the directory of the file; for a folder with multiple files, set to None
SINGLE_FILE = None # for a single file, set to the path of the file; for a folder with multiple files, set to None
COLUMN = "prediction_label" # "Label" for "Useful", "Unhelpful", or "Invalid" or "hallucinated" for "yes" (Hallucinated) or "no"(Not Hallucinated)
SAVE_DIRECTORY = f"MODEL_{COLUMN}" if "model" in FILE_DIRECTORY else f"PROMPT_{COLUMN}"


# Count how often each prediction_label occurs for each actual label
def count_prediction_per_label(file_df):
    # Create a count table: rows = true labels, columns = prediction labels
    count_df = pd.crosstab(file_df['Label'], file_df['prediction_label'])
    
    # Ensure all expected labels and prediction labels are present
    expected_labels = ['Useful', 'Invalid', 'Unhelpful']
    expected_preds = ['good', 'bad']
    for label in expected_labels:
        if label not in count_df.index:
            count_df.loc[label] = 0
    for pred in expected_preds:
        if pred not in count_df.columns:
            count_df[pred] = 0

    # Reorder for consistency
    count_df = count_df.loc[expected_labels, expected_preds]
    return count_df.sort_index()


# Generate an average prompt descriptions per label per model in a file
def average_prediction_counts(filing, all_dfs, model_dir):
    combined = pd.concat(all_dfs)
    averaged = combined.groupby(combined.index).mean()
    filing.save_file(f"{model_dir}_label_prediction_avg.xlsx", averaged, index=True)


# This function summarises on prompt type per model
def analyse_prediction_counts(filing, file):
    df = filing.open_file(file)

    # Drop rows with missing prediction labels since one third of the CQs have been evaluated
    df = df.dropna(subset=['prediction_label'])
    df = df[df['prediction_label'].str.strip() != ""]

    return count_prediction_per_label(df)


# Main function to process all models and their prompts
def main():
    filing = Filing(FILE_DIRECTORY, SINGLE_FILE, SAVE_DIRECTORY)
    model_dirs = filing.file_extraction()

    for model_dir in model_dirs:
        count_dfs = []
        files = filing.file_extraction(model_dir)
        for file in files:
            count_df = analyse_prediction_counts(filing, file)
            count_dfs.append(count_df)
        average_prediction_counts(filing, count_dfs, model_dir)

    print("All models processed successfully." if "model" in FILE_DIRECTORY else "All prompts processed successfully.")


if __name__ == "__main__":
    main()