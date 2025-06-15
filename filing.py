import pandas as pd
import json
import os

class Filing():
    def __init__(self, file_directory, single_file, save_directory):
        self.file_directory = file_directory
        self.single_file = single_file
        self.save_directory = save_directory


    # Return subdirectories if they exist, else return files in the directory
    def file_extraction(self, path: str = None) -> list[str]:
        if self.single_file:
            if not os.path.isfile(self.single_file):
                raise FileNotFoundError(f"{self.single_file} is not a valid file.")
            return [self.single_file]
        path = path or self.file_directory  # fallback to instance value
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a valid directory.")

        entries = [os.path.join(path, entry) for entry in os.listdir(path)]
        directories = [entry for entry in entries if os.path.isdir(entry)]
        files = [entry for entry in entries if os.path.isfile(entry)]

        return directories if directories else files


    # Open a file and return its content as a DataFrame or JSON object
    @staticmethod
    def open_file(file):
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found.")
        if file.endswith('.json'):
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)
        elif file.endswith('.xlsx'):
            return pd.read_excel(file)
        elif file.endswith('.csv'):
            return pd.read_csv(file)
        else:
            raise ValueError(f"Unsupported file format: {file}")        


    # Save the DataFrame with labels to an Excel file
    def save_file(self, input_filename, df, index=False, prefix=""):
        os.makedirs(f"{self.save_directory}", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        output_excel = f"{self.save_directory}/{prefix}{base_name}.xlsx"
        df.to_excel(output_excel, index=index)
        print(f"Saved: {output_excel}")
