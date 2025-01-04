import pandas as pd
from app.reconstruction import unwindow_data

def load_csv(file_path, headers=False, parse_first_column_as_date=False):
    """
    Load CSV data with optional parsing of the first column as dates.

    Parameters:
    - file_path (str): Path to the CSV file.
    - headers (bool): Indicates if the CSV has headers.
    - parse_first_column_as_date (bool): Whether to parse the first column as dates.

    Returns:
    - pd.DataFrame: Loaded and processed data.
    """
    try:
        if parse_first_column_as_date:
            if headers:
                data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
            else:
                data = pd.read_csv(file_path, header=None, sep=',', parse_dates=[0], dayfirst=True)
                if pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
                    data.columns = ['date'] + [f'col_{i-1}' for i in range(1, len(data.columns))]
                    data.set_index('date', inplace=True)
                else:
                    data.columns = [f'col_{i}' for i in range(len(data.columns))]
        else:
            if headers:
                data = pd.read_csv(file_path, sep=',', dayfirst=True)
            else:
                data = pd.read_csv(file_path, header=None, sep=',', dayfirst=True)
                data.columns = [f'col_{i}' for i in range(len(data.columns))]

        # Convert all non-date columns to numeric
        for col in data.columns:
            if parse_first_column_as_date and col == 'date':
                continue  # Skip the date column
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
