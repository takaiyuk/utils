import time
from contextlib import contextmanager


@contextmanager
def timer(name: str) -> None:
    """
    Parameters
    ----------
    name : str
        the name of the function that measures time。

    Examples
    --------
    >>> with timer("Process Modeling"):
            modeling()
    """
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")
