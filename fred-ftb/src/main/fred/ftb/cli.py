from fred.cli.interface import AbstractCLI


class CLI(AbstractCLI):

    def version(self) -> str:
        from fred.ftb.version import version

        return version.value
