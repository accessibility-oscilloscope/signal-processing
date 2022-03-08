#!/usr/bin/env python3

import os, sys
import argparse
import syslog

OSCOPE_LENGTH: int = 4000
SIGNAL_LENGTH: int = 480

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-i", metavar="i", dest="input_path", type=str, help="input")
    parser.add_argument("-o", metavar="o", dest="output_path", type=str, help="output")
    args = parser.parse_args()

    try:
        os.mkfifo(args.input_path)
    except OSError as e:
        syslog.syslog("failed to create FIFO")
        pass
    ifp = os.open(args.input_path, os.O_RDONLY)
    ofp = os.open(args.output_path, os.O_WRONLY)

    while True:
        data = bytearray([])
        while len(data) < 4000:
            read_data = os.read(ifp, 1)
            if read_data:
                data += read_data
        os.write(ofp, data[:SIGNAL_LENGTH])
