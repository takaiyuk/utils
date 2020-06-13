import os


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
