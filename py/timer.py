import logging
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str, logger: logging.Logger = None) -> None:
    """
    Parameters
    ----------
    name : str
        the name of the function that measures time。
    logger: logging.Logger
        logger if you want to use. If None, print() will be used.

    Examples
    --------
    >>> with timer("Process Modeling"):
            modeling()
    """
    t0 = time.time()
    yield
    message = f"[{name}] done in {time.time() - t0:.1f} s"
    if logger is not None:
        logger.info(message)
    else:
        print(message)
