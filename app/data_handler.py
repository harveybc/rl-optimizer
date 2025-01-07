# app/data_handler.py

import pandas as pd
from app.logger import get_logger
from app.reconstruction import unwindow_data  # Assuming it's used elsewhere

logger = get_logger(__name__)

def load_csv(file_path: str, headers: bool = False) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame with optional header processing.

    This function reads a CSV file from the specified path, parses the date column if present,
    assigns appropriate column names, sets the date as the index (if applicable), and ensures
    that all non-date columns are numeric. It also logs the headers and the first five rows
    of the loaded dataset for verification.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to be loaded.
    headers : bool, optional
        Indicates whether the CSV file contains headers (default is False).

    Returns
    -------
    pd.DataFrame
        The loaded and processed DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file_path does not exist.
    pd.errors.ParserError
        If there is an error parsing the CSV file.
    Exception
        For any other exceptions that may occur during the loading process.
    """
    logger.debug(f"Starting load_csv with file_path: {file_path}, headers: {headers}")
    try:
        if headers:
            logger.debug("Loading CSV with headers.")
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
            logger.debug(f"CSV loaded successfully with headers. DataFrame shape: {data.shape}")
        else:
            logger.debug("Loading CSV without headers.")
            data = pd.read_csv(file_path, header=None, sep=',', parse_dates=[0], dayfirst=True)
            logger.debug("CSV loaded without headers.")
            
            # Check if the first column is datetime
            first_col_dtype = data.iloc[:, 0].dtype
            logger.debug(f"First column dtype: {first_col_dtype}")
            if pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
                logger.debug("First column is datetime. Assigning 'date' as index.")
                data.columns = ['date'] + [f'col_{i}' for i in range(1, len(data.columns))]
                data.set_index('date', inplace=True)
                logger.debug("Date column set as index.")
            else:
                logger.debug("First column is not datetime. Assigning generic column names.")
                data.columns = [f'col_{i}' for i in range(len(data.columns))]
            
            # Convert non-date columns to numeric
            non_date_columns = data.columns if 'date' not in data.columns else data.columns[1:]
            logger.debug(f"Converting columns to numeric: {list(non_date_columns)}")
            for col in non_date_columns:
                before_conversion = data[col].copy()
                data[col] = pd.to_numeric(data[col], errors='coerce')
                conversion_issues = data[col].isna().sum()
                if conversion_issues > 0:
                    logger.warning(f"Column '{col}' had {conversion_issues} non-numeric values coerced to NaN.")
        
        # Log headers and first five rows
        logger.debug("Logging DataFrame headers and first five rows.")
        logger.info(f"Headers: {data.columns.tolist()}")
        logger.info("First five rows:")
        logger.info(f"\n{data.head()}")
        
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {file_path}. Error: {fnf_error}")
        raise
    except pd.errors.ParserError as parse_error:
        logger.error(f"Error parsing CSV file: {file_path}. Error: {parse_error}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the CSV: {e}")
        raise
    
    logger.debug("load_csv completed successfully.")
    return data

def write_csv(file_path: str, data: pd.DataFrame, include_date: bool = True, headers: bool = True, window_size: int = None) -> None:
    """
    Saves a pandas DataFrame to a CSV file.

    Parameters
    ----------
    file_path : str
        The path where the CSV file will be saved.
    data : pd.DataFrame
        The DataFrame to save.
    include_date : bool, optional
        Indicates whether to include the date column as the index (default is True).
    headers : bool, optional
        Indicates whether to write out column names (default is True).
    window_size : int, optional
        The window size used during data processing (currently unused).

    Raises
    ------
    Exception
        If there is an error during the saving process.
    """
    logger.debug(f"Starting write_csv with file_path: {file_path}, include_date: {include_date}, headers: {headers}, window_size: {window_size}")
    try:
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
            logger.info(f"DataFrame saved successfully to {file_path} with date index.")
        else:
            data.to_csv(file_path, index=False, header=headers)
            logger.info(f"DataFrame saved successfully to {file_path} without date index.")
    except Exception as e:
        logger.error(f"An error occurred while writing the CSV to {file_path}: {e}")
        raise
    logger.debug("write_csv completed successfully.")
