try:
    from pyzui._version import version as __version__
    from pyzui._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")
