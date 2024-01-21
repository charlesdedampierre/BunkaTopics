import logging

import colorlog

# Create a custom logger named "Bunka"
logger = logging.getLogger("Bunka")

# Set the logging level (e.g., INFO, DEBUG, ERROR)
logger.setLevel(logging.DEBUG)

# Define a custom color scheme for the progress bar
custom_color_scheme = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
    "PROGRESS": "light_blue",  # Customize this color for the progress bar
}

# Create a ColorFormatter with the custom color scheme
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - \033[94m%(name)s\033[0m - %(levelname)s - \033[1m%(message)s\033[0m",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors=custom_color_scheme,
    reset=True,  # Reset colors after each log message
)

# Create a stream handler to display log messages on the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add the stream handler to the logger
logger.addHandler(stream_handler)
