from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

JsonDict = Dict[str, Any]


@dataclass(slots=True)
class VisualizationConfig:
    output_dir: str | Path = "./coin_game_visualizations"
    board_size: int = 3
    dpi: int = 150
    figsize: Tuple[float, float] = (8.0, 5.0)


class CoinGameVisualizer:
    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self.config = config or VisualizationConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_episode_rewards(self, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, *, episode_id: int, filepath: Optional[str | Path] = None) -> Path:
        df = self._to_df(rows)
        df = df[df["episode_id"] == episode_id].copy()
        if df.empty:
            raise ValueError(f"No rows for episode_id={episode_id}")
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        for agent, sub in df.sort_values(["agent", "step_id"]).groupby("agent"):
            ax.plot(sub["step_id"], sub["reward"], marker="o", label=str(agent))
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        ax.set_title(f"Episode {episode_id}: reward timeline")
        ax.legend()
        return self._save(fig, filepath or self.output_dir / f"episode_{episode_id}_rewards.png")

    def plot_behavior_heatmap(self, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, *, x_col: str = "distance_agent_to_coin", y_col: str = "distance_opp_to_coin", value_col: str = "realized_defection", filepath: Optional[str | Path] = None) -> Path:
        df = self._to_df(rows)
        pivot = pd.pivot_table(df, index=y_col, columns=x_col, values=value_col, aggfunc="mean", fill_value=0.0)
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        im = ax.imshow(pivot.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{value_col} heatmap")
        fig.colorbar(im, ax=ax)
        return self._save(fig, filepath or self.output_dir / f"heatmap_{value_col}.png")

    def render_board_state(self, step_record: Mapping[str, Any], *, filepath: Optional[str | Path] = None) -> Path:
        semantic = dict(step_record.get("semantic_state", {}))
        fig, ax = plt.subplots(figsize=(5, 5), dpi=self.config.dpi)
        board_size = int(self.config.board_size)
        ax.set_xlim(-0.5, board_size - 0.5)
        ax.set_ylim(board_size - 0.5, -0.5)
        ax.set_xticks(range(board_size))
        ax.set_yticks(range(board_size))
        ax.grid(True)

        red_pos = semantic.get("red_pos")
        blue_pos = semantic.get("blue_pos")
        red_coin_pos = semantic.get("red_coin_pos")
        blue_coin_pos = semantic.get("blue_coin_pos")

        if red_pos and len(red_pos) >= 2:
            ax.scatter(red_pos[1], red_pos[0], marker="s", s=250, label="red_agent")
        if blue_pos and len(blue_pos) >= 2:
            ax.scatter(blue_pos[1], blue_pos[0], marker="o", s=250, label="blue_agent")
        if red_coin_pos and len(red_coin_pos) >= 2:
            ax.scatter(red_coin_pos[1], red_coin_pos[0], marker="*", s=320, label="red_coin")
        if blue_coin_pos and len(blue_coin_pos) >= 2:
            ax.scatter(blue_coin_pos[1], blue_coin_pos[0], marker="P", s=240, label="blue_coin")

        ax.legend(loc="upper right")
        ax.set_title(f"Episode {step_record.get('episode_id')} step {step_record.get('step_id')}")
        return self._save(fig, filepath or self.output_dir / f"episode_{step_record.get('episode_id')}_step_{step_record.get('step_id')}.png")

    def plot_decision_tree(self, tree_model, feature_names: Sequence[str], *, filepath: Optional[str | Path] = None) -> Path:
        fig, ax = plt.subplots(figsize=(16, 8), dpi=self.config.dpi)
        plot_tree(tree_model, feature_names=list(feature_names), filled=True, ax=ax)
        ax.set_title("Decision tree")
        return self._save(fig, filepath or self.output_dir / "decision_tree.png")

    def _to_df(self, rows: Iterable[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        return rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame([dict(r) for r in rows])

    def _save(self, fig, filepath: str | Path) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
