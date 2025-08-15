import os, sys, time, argparse

import numpy as np
import pandas as pd

# The entry point
if __name__ == "__main__" :
    # Parse command line
    parser = argparse.ArgumentParser(
             prog="pRunHH parallel inference for datasets",
             description="It computes inferences using a HH-ML algorithm.",
             epilog="")
    parser.add_argument('-l', '--log_file',      default=f"cvrptw_test10K_18_05_2025.csv",   nargs='?', action="store", type=str, dest="log_file",         help="The path to the log file")
    args = parser.parse_args()


    # Read log
    df = pd.read_csv(args.log_file)
    print("[{:s}] The average gap is {:f} +/- {:f} (avr obj={:f})".format(args.log_file, np.mean(df['gap']), np.std(df['gap'], ddof=1), np.mean(df['algBestCost'])))

