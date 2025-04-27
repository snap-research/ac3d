"""
This script converts our sloppy output format into a more concise dataframe-like one and evaluates against the GT dataframe.
Our example GT file format:
,filepath,label,num_unique_labels
1,/path/to/video/a.mp4,d,1
2,/path/to/video/b.mp4,d,1
3,/path/to/video/c.mp4,d,1
4,/path/to/video/d.mp4,s,1
5,/path/to/video/e.mp4,s,1
"""
import os
import json
import glob

import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def parse_pred_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    rows = []
    for fp, votes in data.items():
        preds = votes.split(',')
        uniq = set(preds)
        # if all predictions are static, label "s", else "d"
        label = "s" if uniq == {"static"} else "d"
        rows.append({"filepath": fp, "label": label, "num_unique_labels": len(uniq)})
    return pd.DataFrame(rows)


def read_predictions(pred_dir):
    # match files with "rank-" in their name ending with .vote.json
    pattern = os.path.join(pred_dir, "*rank-*.vote.json")
    files = glob.glob(pattern)
    if not files:
        raise ValueError("No prediction files found with pattern " + pattern)
    dfs = []
    for f in files:
        dfs.append(parse_pred_file(f))
    return pd.concat(dfs, ignore_index=True)


def read_val_csv(val_csv_path):
    df = pd.read_csv(val_csv_path)
    # Map all non-"s" to "d"
    df["label"] = df["label"].apply(lambda x: "s" if x.strip() == "s" else "d")
    return df[["filepath", "label"]]


def evaluate(pred_df, val_df, pos_label='d'):
    # Merge on filepath; keep only those with ground truth.
    # Only keep the filename in pred/val df
    val_df["filepath"] = val_df["filepath"].apply(lambda x: os.path.basename(x))
    pred_df["filepath"] = pred_df["filepath"].apply(lambda x: os.path.basename(x))
    merged = pd.merge(val_df, pred_df, on="filepath", suffixes=("_gt", "_pred"))
    if merged.empty:
        raise ValueError("No overlapping filepaths between predictions and ground truth.")
    y_true = merged["label_gt"]
    y_pred = merged["label_pred"]
    # Use "d" as the positive class.
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    num_samples = len(y_true)
    return precision, recall, num_samples


def eval_predictions(val_csv, pred_dir, save_path, **eval_kwargs):
    pred_df = read_predictions(pred_dir)
    val_df = read_val_csv(val_csv)
    precision, recall, num_samples = evaluate(pred_df, val_df, **eval_kwargs)
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")

    if save_path is not None:
        pred_df.to_csv(save_path, index=False)
        print("Saved predictions to", save_path)

        # Let's also save precision/recall as json
        metrics = {"precision": precision, "recall": recall, "num_samples": num_samples}
        metrics_path = os.path.splitext(save_path)[0] + ".metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        print("Saved metrics to", metrics_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir", type=str, help="Directory with prediction json files")
    parser.add_argument("val_csv", type=str, help="CSV file with ground truth annotations")
    parser.add_argument('--pos_label', type=str, default='s')
    parser.add_argument("--save_path", type=str, default=None, help="Path to save predictions")
    args = parser.parse_args()
    eval_predictions(args.val_csv, args.pred_dir, args.save_path, pos_label=args.pos_label)
