import os
import sys
import argparse
import pandas as pd

# To import functions from api.model
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.model import (
    train_model_bert,
    train_model_xlnet,
    train_model_roberta,
    train_model_albert,
    predict_bert,
    predict_xlnet,
    predict_roberta,
    predict_albert
)

TRAINERS = {
    "bert": train_model_bert,
    "xlnet": train_model_xlnet,
    "roberta": train_model_roberta,
    "albert": train_model_albert,
}

PREDICTORS = {
    "bert": predict_bert,
    "xlnet": predict_xlnet,
    "roberta": predict_roberta,
    "albert": predict_albert,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--input", required=True, help="CSV file for training or prediction")
    parser.add_argument("--model", required=True, help="bert/xlnet/roberta/albert")
    parser.add_argument("--model_path", help="Path to trained model")
    parser.add_argument("--output", required=True, help="Where to save output")

    return parser.parse_args()

def main():
    args = parse_args()

    # Validate model name early
    if args.mode == "train" and args.model not in TRAINERS:
        raise ValueError(f"Invalid model '{args.model}'. Choose from: {list(TRAINERS.keys())}")

    if args.mode == "predict" and args.model not in PREDICTORS:
        raise ValueError(f"Invalid model '{args.model}'. Choose from: {list(PREDICTORS.keys())}")

    df = pd.read_csv(args.input)

    if args.mode == "train":
        trainer = TRAINERS[args.model]

        result = trainer(df, args.output)
        print(f"MODEL_SAVED_AT: {result}")

    elif args.mode == "predict":
        predictor = PREDICTORS[args.model]

        model_file = args.model_path   
        print(f"Using model file: {model_file}")

        pred_df = predictor(df, model_file)
        pred_df.to_csv(args.output, index=False)
        print(f"PREDICTION_SAVED_AT: {args.output}")

if __name__ == "__main__":
    main()
