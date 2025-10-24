import os

from fred.version import Version


version = Version.from_path(name="fred-ftb", dirpath=os.path.dirname(__file__))
