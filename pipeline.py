#!/usr/bin/env python3

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from pipeline import core

if __name__ == '__main__':
    core.run_headless()
