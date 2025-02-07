import os
import pandas as pd

def load_dataset(file_name, data_dir="data/raw"):
    """
    Load a dataset from the specified directory.
    Args:
        file_name (str): The name of the file to load (e.g., 'train.csv').
        data_dir (str): The directory where the data file is stored.
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the dataset does not contain required columns.
    """
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} not found in {data_dir}!")
    
    df = pd.read_csv(file_path)
    
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")
    
    print(f"Loaded dataset {file_name} with {len(df)} rows and {len(df.columns)} columns.")
    return df
