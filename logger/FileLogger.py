import sys
import os.path as path


class FileLogger():
    def __init__(self,
                 file_name):
        self.origin_out = sys.stdout
        self.origin_err = sys.stderr

        self.base = path.join("result", "log")
        self.target_path = path.join(self.base, file_name)
        self.target = open(self.target_path, 'w')
        self.target.close()

    def on(self):
        self.target = open(self.target_path, 'a')
        sys.stdout = self.target
        sys.stderr = self.target
    
    def off(self):
        sys.stdout = self.origin_out
        sys.stderr = self.origin_err
        self.target.close()
