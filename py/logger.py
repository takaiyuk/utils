import logging


class Logger:
    @staticmethod
    def get() -> logging.Logger:
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        logger = logging.getLogger(__name__)
        return logger
