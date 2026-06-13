"""
NYC Taxi Trip Duration — Main Production Interface
==================================================
Usage:
    python main.py --data datasets/nyc_taxi_trip_duration.csv
"""

import argparse
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "src"))

from src.pipeline import run_evaluation_pipeline
from src.eda import run_eda
from src.preprocessing import load_and_preprocess_data

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Trip Duration Engine")
    parser.add_argument(
        "--data", 
        type=str, 
        default="datasets/nyc_taxi_trip_duration.csv", 
        help="Path to the CSV dataset"
    )
    parser.add_argument(
        "--eda", 
        action="store_true", 
        help="Run EDA plots before running the pipeline"
    )
    args = parser.parse_args()

    if args.eda:
        print("\n[Main] Running exploratory data analysis and updating /output plots...")
        df = load_and_preprocess_data(args.data)
        run_eda(df)

    run_evaluation_pipeline(data_path=args.data)

if __name__ == "__main__":
    main()