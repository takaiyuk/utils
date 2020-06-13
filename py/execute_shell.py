import subprocess


def execute_shell(cmd: str) -> None:
    cmd_split = cmd.split(" ")
    subprocess.run(cmd_split)
