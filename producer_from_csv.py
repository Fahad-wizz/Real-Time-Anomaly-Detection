"""producer_from_csv.py
Stream rows from a CSV file to stdout as JSON lines or optionally to Kafka.
Usage:
  python producer_from_csv.py --csv /mnt/data/Midterm_53_group.csv --rate 0.01 --mode stdout
Modes:
  stdout : prints JSON lines (default)
  kafka  : sends to Kafka topic (requires kafka-python and running Kafka)
"""
import argparse, json, time, sys
import pandas as pd

def stream_stdout(csv_path, rate=0.01):
    for chunk in pd.read_csv(csv_path, chunksize=5000):
        for _, row in chunk.iterrows():
            rec = row.dropna().to_dict()
            # normalize timestamp column
            if 'Time' in rec:
                try:
                    rec['ts'] = float(rec['Time'])
                except:
                    rec['ts'] = time.time()
            else:
                rec['ts'] = time.time()
            print(json.dumps(rec), flush=True)
            time.sleep(rate)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=r"/mnt/data/Midterm_53_group.csv")
    p.add_argument('--rate', type=float, default=0.01)
    p.add_argument('--mode', choices=['stdout','kafka'], default='stdout')
    p.add_argument('--topic', default='netflow')
    args = p.parse_args()

    if args.mode == 'stdout':
        stream_stdout(args.csv, rate=args.rate)
    else:
        print("Kafka mode selected. See README for Kafka producer example.", file=sys.stderr)

if __name__ == '__main__':
    main()
