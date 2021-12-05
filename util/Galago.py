import os
import ast


class Galago:

    def __init__(self, galago_cmd='galago'):
        self.galago_cmd = galago_cmd

    def exec_dict(self, cmd: str) -> dict:
        """
        Runs galago command
        :param cmd: command to run (ex: 'stats --field+text')
        :return: parsed dict from command output
        """
        return ast.literal_eval(self.exec_str(cmd))

    def exec_str(self, cmd: str) -> str:
        """
        Runs galago command
        :return: command output as string
        """
        return self.exec(cmd).read().strip()

    def exec(self, cmd: str):
        return os.popen(f'{self.galago_cmd} {cmd}')