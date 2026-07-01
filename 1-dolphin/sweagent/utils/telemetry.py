import logging
from logging import Logger

from sweagent.utils.log import get_logger


class BracketFormattedLogger:
    def __init__(self, logger: Logger):
        self._logger = logger

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg.replace("{}", "%s"), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg.replace("{}", "%s"), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg.replace("{}", "%s"), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg.replace("{}", "%s"), *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logger.exception(msg.replace("{}", "%s"), *args, **kwargs)

def get_format_logger(name, *, emoji="🐸", level=None):
    """
    Get a logger with a format that prints messages in brackets.
    In swe-agent, it calls log with %s for format, e.g. `logger.info("msg: %s", value)`
    In omni-coder, it calls log with {} for format, e.g. `logger.info("msg: {}", value)`
    """
    wrap = BracketFormattedLogger(get_logger(name, emoji=emoji))
    # we set default level as `INFO`
    wrap._logger.setLevel(level or logging.INFO)
    return wrap


logger = get_format_logger(name="clear-agent", emoji="🐸")
