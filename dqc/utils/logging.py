import logging


class Logger:
    """Logging utility"""

    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def __init__(self, name, verbosity_level="error"):
        self.logger = logging.getLogger(name)
        self.set_level(verbosity_level)

        self._setup_stream_handler()
        self.logger.propagate = False

    def _setup_stream_handler(self):
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_level(self, verbosity_level):
        if verbosity_level in self.levels:
            self.logger.setLevel(verbosity_level)

    def get_level(self):
        return self.logger.getEffectiveLevel()

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"{message}")
