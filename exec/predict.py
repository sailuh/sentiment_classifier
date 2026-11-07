import sys
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model import predict_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    pred_df = predict_model(
        predict_df=df,
        model_saved_path=args.model_path,
        model_task=AutoModelForSequenceClassification,
        model_tokenizer=AutoTokenizer,
        model_name=args.model_name,
    )

    pred_df.to_csv(args.output, index=False)
    print(f"PREDICTION_SAVED_AT: {args.output}")

if __name__ == "__main__":
    main()