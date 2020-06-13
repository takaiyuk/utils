import yaml


class Config:
    @staticmethod
    def load_yaml(path: str = "./config.yml") -> dict:
        with open(path) as f:
            params = yaml.load(f)
        return params
