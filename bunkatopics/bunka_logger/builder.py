from .config import LoggerConfig


def build_logger():
    import sys

    from loguru import logger

    cfg = LoggerConfig()
    logger.remove()

    logger.add(sys.stderr, format=cfg.fmt, colorize=not cfg.no_color)
    return logger


logger = build_logger()
