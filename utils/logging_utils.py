import os
import os.path as osp
import sys


def mkdir_if_missing(directory):
    if directory and not osp.exists(directory):
        os.makedirs(directory)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, "w")

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def isatty(self):
        if hasattr(self.console, "isatty"):
            try:
                return bool(self.console.isatty())
            except Exception:
                return False
        return False

    def close(self):
        if self.file is not None:
            self.file.close()
