#!/usr/bin/env python

__all__ = ['main']

import os
import sys
import argparse

def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog='',
        description='',
        epilog='',
    )
    parser.add_argument('input', help="")
    parser.add_argument('file', nargs='?', help="")
    parser.add_argument('-s', '--sep', type=str, default=',')
    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args(argv)

    if args.file:
        file = open(args.file)
    else:
        file = sys.stdin


if __name__ == '__main__':
    main(sys.argv[1:])
