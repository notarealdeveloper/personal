#!/usr/bin/env python

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("arguments")
args = parser.parse_args(sys.argv[1:])

input = sys.stdin.read()
