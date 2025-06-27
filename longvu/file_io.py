from iopath.common.file_io import HTTPURLHandler, PathManager as PathManagerBase

__all__ = ["PathManager"]


PathManager = PathManagerBase()
PathManager.register_handler(HTTPURLHandler())
