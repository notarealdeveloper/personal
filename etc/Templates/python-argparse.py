#!/usr/bin/env python

import os
import sys
import argparse

def parse_command_line(args=None):

    if args is None:
        if 'ipython' in os.path.basename(sys.argv[0]):
            args = []
        else:
            args = sys.argv[1:]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "arguments",
        nargs = '*',
        help = "Positional Arguments."
    )

    parser.add_argument(
        "-i",
        "--input-file",
        type = str,
        help = "Input file."
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action = 'store_true',
        help = "Dry run."
    )

    config = parser.parse_args(args)

    return config


def main():

    config = parse_command_line()

    print(config)


if __name__ == "__main__":
    main()
