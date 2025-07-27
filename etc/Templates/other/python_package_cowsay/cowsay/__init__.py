#!/usr/bin/env python3

class _Cowsay:

    def __call__(self, message):
        print(self.get_cow(message))

    def get_cow(self, message):
        import os
        with os.popen(f"cowsay {message}") as fp:
            return fp.read()

import sys
sys.modules['cowsay'] = _Cowsay()
del sys
