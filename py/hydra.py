import hydra


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
