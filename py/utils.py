import logging
import os
import subprocess
from typing import Any

import hydra
import joblib
import yaml


class Config:
    @staticmethod
    def load_yaml(path: str = "./config.yml") -> dict:
        with open(path) as f:
            params = yaml.load(f)
        kf_meth = params["kfold"]["meth"]
        objective = params["objective"]
        assert kf_meth in ["normal", "stratified", "group", "stratified_group"]
        assert objective in ["clf", "reg"]
        return params


class Hydra:
    @staticmethod
    def get_session_id() -> str:
        """hydra cwd: ${project_path}/outputs/YYYY-mm-dd/HH-MM-SS"""
        hydra_cwd = os.getcwd()
        session_id = "-".join(hydra_cwd.split("/")[-2:])
        return session_id

    @staticmethod
    def get_original_cwd() -> str:
        try:
            return hydra.utils.get_original_cwd()
        except AttributeError:
            return "."


class Jbl:
    @staticmethod
    def load(path: str) -> Any:
        return joblib.load(path)

    @staticmethod
    def save(obj: Any, path: str) -> None:
        joblib.dump(obj, path, compress=3)


class Logger:
    @staticmethod
    def get() -> logging.Logger:
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        logger = logging.getLogger(__name__)
        return logger


def execute_shell(cmd: str) -> None:
    cmd_split = cmd.split(" ")
    subprocess.run(cmd_split)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
