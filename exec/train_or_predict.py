import sys
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model import predict_model, train_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_path", help="Path to trained model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", required=True, help="Hugging Face model name, e.g., 'bert-base-cased'")
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    if args.mode == "train":
        saved_model_path = train_model(
            train_df=df,
            model_save_path=args.output,
            model_task=AutoModelForSequenceClassification,
            model_tokenizer=AutoTokenizer,
            model_name=args.model_name
        )
        print(f"MODEL_SAVED_AT: {saved_model_path}")

    elif args.mode == "predict":
        pred_df = predict_model(
            predict_df=df,
            model_saved_path=args.model_path,
            model_task=AutoModelForSequenceClassification,
            model_tokenizer=AutoTokenizer,
            model_name=args.model_name
        )
        pred_df.to_csv(args.output, index=False)
        print(f"PREDICTION_SAVED_AT: {args.output}")

if __name__ == "__main__":
    main()
