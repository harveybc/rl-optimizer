import pandas as pd
from app.reconstruction import unwindow_data

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_csv(file_path, headers=False):
    """
    Loads a CSV file into a pandas DataFrame with optional header processing.

    This function reads a CSV file from the specified path, parses the date column if present,
    assigns appropriate column names, sets the date as the index (if applicable), and ensures
    that all non-date columns are numeric. It also prints the headers and the first five rows
    of the loaded dataset for verification.

    Parameters:
    ----------
    file_path : str
        The path to the CSV file to be loaded.
    headers : bool, optional
        Indicates whether the CSV file contains headers (default is False).

    Returns:
    -------
    pd.DataFrame
        The loaded and processed DataFrame.

    Raises:
    ------
    FileNotFoundError
        If the specified file_path does not exist.
    pd.errors.ParserError
        If there is an error parsing the CSV file.
    Exception
        For any other exceptions that may occur during the loading process.
    """
    logging.debug(f"Starting load_csv with file_path: {file_path}, headers: {headers}")
    try:
        if headers:
            logging.debug("Loading CSV with headers.")
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
            logging.debug(f"CSV loaded successfully with headers. DataFrame shape: {data.shape}")
        else:
            logging.debug("Loading CSV without headers.")
            data = pd.read_csv(file_path, header=None, sep=',', parse_dates=[0], dayfirst=True)
            logging.debug("CSV loaded without headers.")
            
            # Check if the first column is datetime
            first_col_dtype = data.iloc[:, 0].dtype
            logging.debug(f"First column dtype: {first_col_dtype}")
            if pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
                logging.debug("First column is datetime. Assigning 'date' as index.")
                data.columns = ['date'] + [f'col_{i}' for i in range(1, len(data.columns))]
                data.set_index('date', inplace=True)
                logging.debug("Date column set as index.")
            else:
                logging.debug("First column is not datetime. Assigning generic column names.")
                data.columns = [f'col_{i}' for i in range(len(data.columns))]
            
            # Convert non-date columns to numeric
            non_date_columns = data.columns if 'date' not in data.columns else data.columns[1:]
            logging.debug(f"Converting columns to numeric: {list(non_date_columns)}")
            for col in non_date_columns:
                before_conversion = data[col].copy()
                data[col] = pd.to_numeric(data[col], errors='coerce')
                conversion_issues = data[col].isna().sum()
                if conversion_issues > 0:
                    logging.warning(f"Column '{col}' had {conversion_issues} non-numeric values coerced to NaN.")
        
        # Print headers and first five rows
        logging.debug("Printing DataFrame headers and first five rows.")
        print("Headers:", data.columns.tolist())
        print("First five rows:")
        print(data.head())
        
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {file_path}. Error: {fnf_error}")
        raise
    except pd.errors.ParserError as parse_error:
        logging.error(f"Error parsing CSV file: {file_path}. Error: {parse_error}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the CSV: {e}")
        raise
    
    logging.debug("load_csv completed successfully.")
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
