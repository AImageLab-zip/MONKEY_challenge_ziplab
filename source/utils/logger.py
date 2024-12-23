import logging
import os
import time


def get_logger(name, args={}):
    """
    Get a logger with the specified name and configuration.

    Args:
        name (str): The name of the logger.
        args (object): An object containing the configuration options.

    Returns:
        logging.Logger: The configured logger instance.
    """

    # Use getattr to access attributes with a default fallback
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    log_dir = getattr(args, "log_dir", "logs")
    console_log = getattr(args, "console_log", True)
    file_log = getattr(args, "file_log", False)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    # set name and level of logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers set up
    if logger.hasHandlers():
        logger.handlers.clear()  # Clear existing handlers if re-setting up the logger

    # if console_log set the logger to log on the terminal like a print statement
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # if file logging has been enabled, also logs to a .log file in a logdir specified folder
    if file_log:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{name}_{timestamp}_{'_debug' if getattr(args, 'debug', False) else ''}.log"
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
