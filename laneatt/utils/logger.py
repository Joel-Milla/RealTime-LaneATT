import logging
import os

def setup_logging(logs_dir):
    """
        Setups the logging configuration for the project.

        Returns:
            logger: The logger object.
    """

    # Create logs directory if not exists
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Creates the latest log file
    log_files = os.listdir(logs_dir)
    log_files = [f for f in log_files if f.endswith(".log")]
    log_files.sort()
    if len(log_files) > 0:
        number = int(log_files[-1].split("_")[1].split(".")[0])
        number += 1
        log_path = os.path.join(logs_dir, "train_{:04d}.log".format(number))
    else:
        log_path = os.path.join(logs_dir, "train_0000.log")

    # Specifies the format of the log messages and returns the logger
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])

    return logging.getLogger(__name__)