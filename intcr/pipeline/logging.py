import logging
import os
from datetime import datetime


global _CURRENT_LOGGER
_CURRENT_LOGGER = None


class Logger:
    def __init__(self, logger_name, root):
        self._logger2file = logger_name + '_file'
        self._logger2stderr = logger_name + '_stderr'

        self._logger_file = logging.getLogger(self._logger2file)
        self._logger_file.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(root, datetime.now().strftime("%m%d%Y%H%M%S")))
        self._logger_file.addHandler(fh)

        self._logger_stderr = logging.getLogger(self._logger2stderr)
        self._logger_stderr.addHandler(logging.StreamHandler())

    def set_stderr_level(self, level):
        self._logger_stderr.setLevel(level)

    def debug(self, msg):
        self._logger_stderr.debug(msg)
        self._logger_file.debug(msg)

    def info(self, msg):
        self._logger_stderr.info(msg)
        self._logger_file.info(msg)

    def warning(self, msg):
        self._logger_stderr.warning(msg)
        self._logger_file.warning(msg)

    def error(self, msg):
        self._logger_stderr.error(msg)
        self._logger_file.error(msg)

    def critical(self, msg):
        self._logger_stderr.critical(msg)
        self._logger_file.critical(msg)
