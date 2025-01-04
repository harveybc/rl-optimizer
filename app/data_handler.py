import pandas as pd
from app.reconstruction import unwindow_data

import pandas as pd

def load_csv(file_path, headers=False):
    """
    Load CSV data where the first column represents tick numbers.

    Parameters:
    - file_path (str): Path to the CSV file.
    - headers (bool): Indicates if the CSV has headers.

    Returns:
    - pd.DataFrame: Loaded and processed data with the first column as 'tick'.
    """
    try:
        if headers:
            data = pd.read_csv(file_path, sep=',', dayfirst=True)
            data.rename(columns={data.columns[0]: 'tick'}, inplace=True)
        else:
            data = pd.read_csv(file_path, header=None, sep=',', dayfirst=True)
            # Rename the first column to 'tick' for clarity
            data.rename(columns={0: 'tick'}, inplace=True)
            # Rename other columns as 'col_1', 'col_2', etc.
            data.rename(columns={i: f'col_{i}' for i in range(1, len(data.columns))}, inplace=True)

        # Convert the 'tick' column to integer or float as needed
        data['tick'] = pd.to_numeric(data['tick'], errors='coerce').fillna(0).astype(int)

        # Convert all other columns to numeric, filling non-convertible values with 0
        for col in data.columns:
            if col != 'tick':
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        raise

    return data


def write_csv(file_path, data, include_date=True, headers=True, window_size=None):
    try:
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
        else:
            data.to_csv(file_path, index=False, header=headers)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise
