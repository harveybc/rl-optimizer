import sys
import traceback
import logging

# Setup logging to a file to capture sys.path changes
logging.basicConfig(filename='sys_path_log.txt', level=logging.INFO)

def log_sys_path_change():
    logging.info("sys.path changed:")
    for path in sys.path:
        logging.info(f" - {path}")
    logging.info("Stack trace of modification:")
    logging.info(''.join(traceback.format_stack()))

# Wrap sys.path in a custom list to track changes
class PathWatcher(list):
    def append(self, item):
        super().append(item)
        log_sys_path_change()
    
    def insert(self, index, item):
        super().insert(index, item)
        log_sys_path_change()

    def __setitem__(self, index, item):
        super().__setitem__(index, item)
        log_sys_path_change()

# Replace sys.path with the custom watcher
sys.path = PathWatcher(sys.path)

# Example code to test sys.path modifications
sys.path.append("new_test_path")  # This will trigger a log entry
