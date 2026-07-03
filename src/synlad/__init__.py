from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("synlad")
except PackageNotFoundError:
    __version__ = "unknown version"
