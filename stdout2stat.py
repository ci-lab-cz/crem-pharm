#!/usr/bin/env python3

import argparse
import pandas as pd


def convert(input_fname, output_fname):

    output = []
    res = {}

    with open(input_fname) as f:

        f.readline()  # skip the first line
        for line in f:
            # skip initial screening
            if line.strip() == '===== Initial screening =====':
                f.readline()
                continue
            if line.startswith('====='):
                if res:
                    output.append(res)
                res = {'parent_mol': line.split(' ')[1].strip()}
            elif not line.strip() or line.startswith('search deep'):
                continue
            elif line.startswith('preprocessing'):
                res['preprocessing'] = line.split(' ')[1].strip()
            elif line.startswith('overall time'):
                res['overall time'] = line.split(': ')[1].strip()
            else:
                name = line.split(':')[0].strip()
                value = line.split(', ')[1].strip()
                res[name] = value

    if output:
        df = pd.DataFrame(output)
        df.to_csv(output_fname, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Parse input stdout logs and create a table with timings of '
                                                 'each step.')
    parser.add_argument('-i', '--input', metavar='FILENAME', required=True, type=str,
                        help='input STDOUT file.')
    parser.add_argument('-o', '--output', metavar='FILENAME', required=True, type=str,
                        help='output text file.')
    args = parser.parse_args()

    convert(args.input, args.output)


if __name__ == '__main__':
    main()
