from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ACTION_LABELS = {
    0: "class_0",
    1: "class_1",
    2: "class_2",
    3: "class_3",
    4: "class_4",
}


def summarize_rules(rules_csv: str | Path, top_k: int = 10):
    rules = pd.read_csv(rules_csv)

    feature_counter = Counter()
    for rule in rules["rule"]:
        names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|>)", str(rule))
        feature_counter.update(names)

    top_rules = (
        rules.sort_values(["purity", "sample_count"], ascending=[False, False])
        .head(top_k)
        .copy()
    )
    top_rules["predicted_action_label"] = top_rules["predicted_class"].map(ACTION_LABELS).fillna("unknown")

    by_class = (
        rules.groupby("predicted_class")
        .agg(
            num_rules=("leaf_id", "count"),
            total_coverage=("sample_count", "sum"),
            mean_purity=("purity", "mean"),
            max_purity=("purity", "max"),
        )
        .reset_index()
        .sort_values("predicted_class")
    )

    feature_df = pd.DataFrame(feature_counter.most_common(), columns=["feature", "count"])
    return top_rules, by_class, feature_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-csv", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-top-rules", default="top_rules.csv")
    parser.add_argument("--out-by-class", default="rules_by_class.csv")
    parser.add_argument("--out-feature-frequency", default="rule_feature_frequency.csv")
    args = parser.parse_args()

    top_rules, by_class, feature_df = summarize_rules(args.rules_csv, top_k=args.top_k)
    top_rules.to_csv(args.out_top_rules, index=False)
    by_class.to_csv(args.out_by_class, index=False)
    feature_df.to_csv(args.out_feature_frequency, index=False)

    print("Top rules saved to:", Path(args.out_top_rules).resolve())
    print("Rules by class saved to:", Path(args.out_by_class).resolve())
    print("Feature frequency saved to:", Path(args.out_feature_frequency).resolve())
    print("\nTop rules:")
    print(top_rules[["leaf_id", "predicted_class", "predicted_action_label", "sample_count", "purity", "rule"]].to_string(index=False))
    print("\nRules by class:")
    print(by_class.to_string(index=False))
    print("\nMost frequent rule features:")
    print(feature_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
