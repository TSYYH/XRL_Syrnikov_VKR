from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def compute_behavior_metrics(per_agent_csv: str | Path, episode_csv: str | Path):
    per_agent = pd.read_csv(per_agent_csv)
    episode = pd.read_csv(episode_csv)
    metrics = []

    for agent, sub in per_agent.groupby("agent"):
        total_steps = len(sub)
        picked_any = int(sub["picked_any_coin"].fillna(False).astype(bool).sum())
        picked_own = int(sub["picked_own_coin"].fillna(False).astype(bool).sum())
        picked_opp = int(sub["picked_opp_coin"].fillna(False).astype(bool).sum())
        picked_red = int(sub["picked_red_coin"].fillna(False).astype(bool).sum()) if "picked_red_coin" in sub.columns else None
        picked_blue = int(sub["picked_blue_coin"].fillna(False).astype(bool).sum()) if "picked_blue_coin" in sub.columns else None
        mean_reward = float(sub["reward"].mean())

        cooperation_rate = picked_own / total_steps if total_steps else 0.0
        defection_rate = picked_opp / total_steps if total_steps else 0.0
        own_share = picked_own / picked_any if picked_any else 0.0
        opp_share = picked_opp / picked_any if picked_any else 0.0
        cooperativity_index = own_share - opp_share

        row = {
            "agent": agent,
            "mean_reward_per_step": mean_reward,
            "total_steps": total_steps,
            "picked_any_coin": picked_any,
            "picked_own_coin": picked_own,
            "picked_opp_coin": picked_opp,
            "cooperation_rate": cooperation_rate,
            "defection_rate": defection_rate,
            "share_own_coin": own_share,
            "share_opp_coin": opp_share,
            "cooperativity_index": cooperativity_index,
        }
        if picked_red is not None:
            row["picked_red_coin"] = picked_red
        if picked_blue is not None:
            row["picked_blue_coin"] = picked_blue
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics).sort_values("agent").reset_index(drop=True)
    reward_cols = [c for c in episode.columns if c.startswith("reward_")]
    if len(reward_cols) >= 2:
        ep = episode.copy()
        ep["reward_asymmetry"] = (ep[reward_cols[0]] - ep[reward_cols[1]]).abs()
        asymmetry_mean = float(ep["reward_asymmetry"].mean())
        collective_return = float(ep[reward_cols].sum(axis=1).mean())
    else:
        asymmetry_mean = None
        collective_return = None

    summary = pd.DataFrame([{
        "mean_collective_return_per_episode": collective_return,
        "mean_reward_asymmetry_per_episode": asymmetry_mean,
        "num_episodes": int(len(episode)),
    }])
    return metrics_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-agent-csv", required=True)
    parser.add_argument("--episode-csv", required=True)
    parser.add_argument("--out-metrics", default="behavior_metrics.csv")
    parser.add_argument("--out-summary", default="behavior_summary.csv")
    args = parser.parse_args()

    metrics_df, summary_df = compute_behavior_metrics(args.per_agent_csv, args.episode_csv)
    metrics_df.to_csv(args.out_metrics, index=False)
    summary_df.to_csv(args.out_summary, index=False)

    print("Behavior metrics saved to:", Path(args.out_metrics).resolve())
    print("Behavior summary saved to:", Path(args.out_summary).resolve())
    print(metrics_df.to_string(index=False))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
