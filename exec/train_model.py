import sys
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model import train_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    
    saved_model_path = train_model(
        train_df=df,
        model_save_path=args.output,
        model_task=AutoModelForSequenceClassification,
        model_tokenizer=AutoTokenizer,
        model_name=args.model_name,
    )

    print(f"MODEL_SAVED_AT: {saved_model_path}")

if __name__ == "__main__":
    main()