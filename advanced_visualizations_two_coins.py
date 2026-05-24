
from __future__ import annotations

"""
advanced_visualizations.py

Расширенный модуль визуализации для Coin Game с двухмонетной логикой.

Идея модуля соответствует методике, в которой визуализация выполняет
аналитическую, а не только иллюстративную функцию. Поддерживаются пять
типов визуализаций:

1. Визуализация траекторий:
   - последовательность кадров эпизода;
   - агрегированная схема перемещений агентов и монет.

2. Временные диаграммы событий:
   - действия, награды;
   - подбор своей / чужой / красной / синей монеты;
   - сопоставление динамики двух агентов.

3. Тепловые карты и карты частот:
   - посещаемость клеток;
   - позиции кооперации и дефекции;
   - распределение действий относительно своей и чужой монеты.

4. Визуализация суррогатной модели:
   - таблица top-k правил;
   - частота использования признаков в правилах.

5. Сопоставление правил с траекториями:
   - для шагов эпизода показывается, какое правило было активировано.

Скрипт рассчитан на работу с выгрузками:
- coin_game_steps.jsonl
- coin_game_per_agent.csv
- coin_game_features.csv
- coin_game_rules.csv
- rule_feature_frequency.csv (необязательно)
"""

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTION_LABELS = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


# =========================================================
# ----------------------- I/O helpers ---------------------
# =========================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_steps_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def maybe_int(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return None


def to_number(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except Exception:
        text = str(value).strip().lower()
        if text == "true":
            return 1.0
        if text == "false":
            return 0.0
        return np.nan


def get_pair(value: Any) -> Tuple[Optional[int], Optional[int]]:
    if value is None:
        return (None, None)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return (None, None)
    try:
        seq = list(value)
    except Exception:
        return (None, None)
    if len(seq) < 2:
        return (None, None)
    return maybe_int(seq[0]), maybe_int(seq[1])


# =========================================================
# -------------------- state extraction -------------------
# =========================================================

def extract_episode_steps(steps: Sequence[Dict[str, Any]], episode_id: int) -> List[Dict[str, Any]]:
    out = [s for s in steps if int(s.get("episode_id", -1)) == int(episode_id)]
    out.sort(key=lambda x: int(x.get("step_id", 0)))
    return out


def get_decision_state(step_record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(step_record.get("decision_state") or step_record.get("semantic_state") or {})


def extract_board_entities(state: Dict[str, Any]) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """
    Возвращает позиции двух агентов и двух монет.
    Поддерживает варианты red/blue и резервные альтернативные имена полей.
    """
    red_pos = get_pair(state.get("red_pos", state.get("agent0_pos")))
    blue_pos = get_pair(state.get("blue_pos", state.get("agent1_pos")))

    red_coin_pos = get_pair(state.get("red_coin_pos"))
    blue_coin_pos = get_pair(state.get("blue_coin_pos"))

    # fallback на случай, если в state присутствует только агрегированная монета
    if red_coin_pos == (None, None):
        rc_row = maybe_int(state.get("red_coin_row"))
        rc_col = maybe_int(state.get("red_coin_col"))
        if rc_row is not None and rc_col is not None:
            red_coin_pos = (rc_row, rc_col)

    if blue_coin_pos == (None, None):
        bc_row = maybe_int(state.get("blue_coin_row"))
        bc_col = maybe_int(state.get("blue_coin_col"))
        if bc_row is not None and bc_col is not None:
            blue_coin_pos = (bc_row, bc_col)

    return {
        "red_pos": red_pos,
        "blue_pos": blue_pos,
        "red_coin_pos": red_coin_pos,
        "blue_coin_pos": blue_coin_pos,
    }


# =========================================================
# 1. Visualization of trajectories
# =========================================================

def plot_episode_frames_two_coins(
    steps: Sequence[Dict[str, Any]],
    episode_id: int,
    out_path: str | Path,
    board_size: int = 3,
    max_frames: int = 12,
) -> Path:
    episode_steps = extract_episode_steps(steps, episode_id)
    if not episode_steps:
        raise ValueError(f"No steps for episode_id={episode_id}")

    if len(episode_steps) <= max_frames:
        chosen = episode_steps
    else:
        idxs = np.linspace(0, len(episode_steps) - 1, max_frames).round().astype(int)
        chosen = [episode_steps[i] for i in idxs]

    n = len(chosen)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, step in zip(axes, chosen):
        state = get_decision_state(step)
        entities = extract_board_entities(state)
        red_pos = entities["red_pos"]
        blue_pos = entities["blue_pos"]
        red_coin_pos = entities["red_coin_pos"]
        blue_coin_pos = entities["blue_coin_pos"]

        ax.set_xlim(-0.5, board_size - 0.5)
        ax.set_ylim(board_size - 0.5, -0.5)
        ax.set_xticks(range(board_size))
        ax.set_yticks(range(board_size))
        ax.grid(True)
        ax.axis("on")

        if red_pos != (None, None):
            ax.scatter(red_pos[1], red_pos[0], marker="s", s=260, label="red_agent")
        if blue_pos != (None, None):
            ax.scatter(blue_pos[1], blue_pos[0], marker="o", s=260, label="blue_agent")
        if red_coin_pos != (None, None):
            ax.scatter(red_coin_pos[1], red_coin_pos[0], marker="*", s=360, label="red_coin")
        if blue_coin_pos != (None, None):
            ax.scatter(blue_coin_pos[1], blue_coin_pos[0], marker="P", s=260, label="blue_coin")

        step_id = step.get("step_id")
        actions = step.get("actions", {})
        ax.set_title(
            f"ep {episode_id}, t={step_id}\n"
            f"a0={ACTION_LABELS.get(actions.get('0'), actions.get('0'))}, "
            f"a1={ACTION_LABELS.get(actions.get('1'), actions.get('1'))}"
        )

    fig.suptitle(
        f"Coin Game: последовательность кадров эпизода {episode_id}",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_episode_trajectory_overlay_two_coins(
    steps: Sequence[Dict[str, Any]],
    episode_id: int,
    out_path: str | Path,
    board_size: int = 3,
) -> Path:
    episode_steps = extract_episode_steps(steps, episode_id)
    if not episode_steps:
        raise ValueError(f"No steps for episode_id={episode_id}")

    red_agent_path = []
    blue_agent_path = []
    red_coin_path = []
    blue_coin_path = []

    for step in episode_steps:
        state = get_decision_state(step)
        entities = extract_board_entities(state)
        red_agent_path.append(entities["red_pos"])
        blue_agent_path.append(entities["blue_pos"])
        red_coin_path.append(entities["red_coin_pos"])
        blue_coin_path.append(entities["blue_coin_pos"])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(board_size - 0.5, -0.5)
    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))
    ax.grid(True)

    def draw_path(path: Sequence[Tuple[Optional[int], Optional[int]]], label: str, marker: str = "o") -> None:
        xs = [p[1] for p in path if p != (None, None)]
        ys = [p[0] for p in path if p != (None, None)]
        if xs and ys:
            ax.plot(xs, ys, marker=marker, label=label)
            for i, p in enumerate(path):
                if p != (None, None):
                    ax.text(p[1] + 0.03, p[0] + 0.03, str(i), fontsize=8)

    draw_path(red_agent_path, "red_agent_path", marker="o")
    draw_path(blue_agent_path, "blue_agent_path", marker="o")

    for coin_pos in red_coin_path:
        if coin_pos != (None, None):
            ax.scatter(coin_pos[1], coin_pos[0], marker="*", s=180, label=None)

    for coin_pos in blue_coin_path:
        if coin_pos != (None, None):
            ax.scatter(coin_pos[1], coin_pos[0], marker="P", s=120, label=None)

    ax.set_title(f"Траектории агентов и позиции монет в эпизоде {episode_id}")
    ax.legend()
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================
# 2. Event timelines
# =========================================================

def plot_event_timeline_two_coins(
    per_agent_df: pd.DataFrame,
    episode_id: int,
    agent: str,
    out_path: str | Path,
) -> Path:
    """
    Временная диаграмма для одного агента:
    - действия;
    - награды;
    - подбор own/opp/red/blue coin.
    """
    df = per_agent_df.copy()
    df["agent"] = df["agent"].astype(str)
    df = df[(df["episode_id"] == episode_id) & (df["agent"] == str(agent))].copy()
    df = df.sort_values("step_id")

    if df.empty:
        raise ValueError(f"No rows for episode_id={episode_id}, agent={agent}")

    steps = df["step_id"].values
    actions = df["action"].values
    rewards = df["reward"].values

    fig, ax1 = plt.subplots(figsize=(13, 5), dpi=150)

    ax1.plot(steps, actions, marker="o", label="action")
    ax1.set_xlabel("Шаг")
    ax1.set_ylabel("Действие")
    ax1.set_title(f"Временная диаграмма событий: episode={episode_id}, agent={agent}")
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels([ACTION_LABELS[i] for i in range(5)])

    event_specs = [
        ("picked_own_coin", "s", 100),
        ("picked_opp_coin", "x", 100),
        ("picked_red_coin", "^", 100),
        ("picked_blue_coin", "v", 100),
    ]
    for col, marker, size in event_specs:
        if col in df.columns:
            mask = df[col].fillna(False).astype(bool).values
            if mask.any():
                ax1.scatter(steps[mask], actions[mask], marker=marker, s=size, label=col)

    damage_mask = rewards < 0
    if np.any(damage_mask):
        ax1.scatter(steps[damage_mask], actions[damage_mask], marker="D", s=80, label="negative_reward")

    ax2 = ax1.twinx()
    ax2.plot(steps, rewards, marker="^", linestyle="--", label="reward")
    ax2.set_ylabel("Награда")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_reciprocity_timeline(
    per_agent_df: pd.DataFrame,
    episode_id: int,
    out_path: str | Path,
    agents: Sequence[str] = ("0", "1"),
) -> Path:
    """
    Двухпанельная временная диаграмма:
    - сверху: награды двух агентов;
    - снизу: события own/opp coin для анализа взаимности.
    """
    df = per_agent_df.copy()
    df["agent"] = df["agent"].astype(str)
    df = df[df["episode_id"] == episode_id].copy()
    df = df.sort_values(["step_id", "agent"])

    if df.empty:
        raise ValueError(f"No rows for episode_id={episode_id}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=150, sharex=True)

    for agent in agents:
        sub = df[df["agent"] == str(agent)].sort_values("step_id")

        axes[0].plot(sub["step_id"], sub["reward"], marker="o", label=f"reward agent {agent}")

        own_mask = sub["picked_own_coin"].fillna(False).astype(bool)
        opp_mask = sub["picked_opp_coin"].fillna(False).astype(bool)

        axes[1].scatter(
            sub["step_id"][own_mask],
            [int(agent)] * int(own_mask.sum()),
            marker="s",
            s=100,
            label=f"own coin agent {agent}",
        )
        axes[1].scatter(
            sub["step_id"][opp_mask],
            [int(agent)] * int(opp_mask.sum()),
            marker="x",
            s=100,
            label=f"opp coin agent {agent}",
        )

    axes[0].set_ylabel("Награда")
    axes[0].set_title(f"Временная динамика наград: эпизод {episode_id}")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Шаг")
    axes[1].set_ylabel("Событие агента")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["agent 0", "agent 1"])
    axes[1].set_title("Кооперация и дефекция во времени")
    axes[1].legend(fontsize=8)

    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================
# 3. Heatmaps and frequency maps
# =========================================================

def build_visit_matrix(df: pd.DataFrame, row_col_prefix: str = "agent", board_size: int = 3) -> np.ndarray:
    mat = np.zeros((board_size, board_size), dtype=float)
    row_name = f"{row_col_prefix}_row"
    col_name = f"{row_col_prefix}_col"

    for _, row in df.iterrows():
        r = maybe_int(row.get(row_name))
        c = maybe_int(row.get(col_name))
        if r is not None and c is not None and 0 <= r < board_size and 0 <= c < board_size:
            mat[r, c] += 1.0
    return mat


def _plot_matrix(mat: np.ndarray, title: str, out_path: str | Path, board_size: int = 3) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(mat, origin="upper", aspect="equal")
    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(title)

    for r in range(board_size):
        for c in range(board_size):
            ax.text(c, r, int(mat[r, c]), ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax)
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_visit_heatmap(
    per_agent_df: pd.DataFrame,
    out_path: str | Path,
    agent: Optional[str] = None,
    board_size: int = 3,
) -> Path:
    df = per_agent_df.copy()
    if agent is not None:
        df["agent"] = df["agent"].astype(str)
        df = df[df["agent"] == str(agent)].copy()

    mat = build_visit_matrix(df, row_col_prefix="agent", board_size=board_size)
    title = "Тепловая карта посещений" + (f" (agent={agent})" if agent is not None else "")
    return _plot_matrix(mat, title, out_path, board_size=board_size)


def plot_pick_heatmap(
    per_agent_df: pd.DataFrame,
    out_path: str | Path,
    pick_kind: str = "own",
    agent: Optional[str] = None,
    board_size: int = 3,
) -> Path:
    """
    pick_kind in {"own", "opp", "red", "blue"}
    """
    col_map = {
        "own": "picked_own_coin",
        "opp": "picked_opp_coin",
        "red": "picked_red_coin",
        "blue": "picked_blue_coin",
    }
    if pick_kind not in col_map:
        raise ValueError(f"Unsupported pick_kind={pick_kind}")

    df = per_agent_df.copy()
    col = col_map[pick_kind]
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in dataframe")

    df = df[df[col].fillna(False).astype(bool)].copy()
    if agent is not None:
        df["agent"] = df["agent"].astype(str)
        df = df[df["agent"] == str(agent)].copy()

    out_path = Path(out_path)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"Для выбранной выборки отсутствуют события\nтипа {pick_kind}",
            ha="center", va="center", fontsize=14,
        )
        ax.set_title(f"Тепловая карта событий типа {pick_kind}")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    mat = build_visit_matrix(df, row_col_prefix="agent", board_size=board_size)
    title = f"Тепловая карта событий типа {pick_kind}" + (f" (agent={agent})" if agent is not None else "")
    return _plot_matrix(mat, title, out_path, board_size=board_size)


def plot_action_frequency_by_coin_relation(
    per_agent_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    df = per_agent_df.copy()
    df["action_label"] = df["action"].map(ACTION_LABELS)

    if "coin_owner_relation" not in df.columns:
        if "coin_owner" in df.columns and "agent" in df.columns:
            df["coin_owner_relation"] = np.where(
                df["coin_owner"].astype(str) == df["agent"].astype(str),
                "self",
                "opp",
            )
        else:
            # fallback: derive relation from pickup events if available
            df["coin_owner_relation"] = np.where(
                df.get("picked_own_coin", False).fillna(False).astype(bool),
                "self",
                np.where(df.get("picked_opp_coin", False).fillna(False).astype(bool), "opp", "unknown")
            )

    grouped = (
        df.groupby(["coin_owner_relation", "action_label"])
        .size()
        .reset_index(name="count")
    )

    relations = [rel for rel in ["self", "opp", "unknown"] if rel in grouped["coin_owner_relation"].astype(str).unique()]
    actions = [ACTION_LABELS[i] for i in sorted(ACTION_LABELS)]
    x = np.arange(len(actions))
    width = 0.25 if len(relations) >= 3 else 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i, rel in enumerate(relations):
        sub = grouped[grouped["coin_owner_relation"] == rel]
        counts = []
        for action in actions:
            row = sub[sub["action_label"] == action]
            counts.append(int(row["count"].iloc[0]) if not row.empty else 0)
        ax.bar(x + i * width, counts, width=width, label=rel)

    tick_shift = width * (len(relations) - 1) / 2 if relations else 0.0
    ax.set_xticks(x + tick_shift)
    ax.set_xticklabels(actions)
    ax.set_xlabel("Действие")
    ax.set_ylabel("Частота")
    ax.set_title("Частота действий при своей и чужой монете")
    ax.legend(title="Тип монеты")
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================
# 4. Surrogate model visualizations
# =========================================================

def plot_top_rules_table(
    rules_df: pd.DataFrame,
    out_path: str | Path,
    top_k: int = 10,
) -> Path:
    df = rules_df.copy()
    sort_cols = [c for c in ["sample_count", "purity", "empirical_accuracy"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    df = df.head(top_k).copy()

    display_cols = [c for c in ["leaf_id", "predicted_class", "sample_count", "purity", "rule"] if c in df.columns]
    df = df[display_cols]

    if "rule" in df.columns:
        df["rule"] = df["rule"].astype(str).map(lambda s: "\n".join(textwrap.wrap(s, width=70)))

    fig_h = max(4, 0.8 * len(df) + 1)
    fig, ax = plt.subplots(figsize=(18, fig_h), dpi=150)
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    ax.set_title("Таблица наиболее значимых правил", pad=20)
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rule_feature_frequency(
    freq_df: pd.DataFrame,
    out_path: str | Path,
    top_k: int = 15,
) -> Path:
    df = freq_df.copy()
    feature_col = next((c for c in ["feature", "feature_name", "name"] if c in df.columns), None)
    count_col = next((c for c in ["count", "frequency", "n"] if c in df.columns), None)
    if feature_col is None or count_col is None:
        raise ValueError("Could not find feature/count columns in rule feature frequency dataframe")

    df = df.sort_values(count_col, ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(df[feature_col].astype(str), df[count_col].astype(float))
    ax.invert_yaxis()
    ax.set_xlabel("Частота использования")
    ax.set_ylabel("Признак")
    ax.set_title("Наиболее часто используемые признаки в правилах")

    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================
# 5. Rule-to-trajectory alignment
# =========================================================

def parse_atomic_condition(text: str) -> Tuple[str, str, float]:
    text = text.strip()
    if "<=" in text:
        left, right = text.split("<=", 1)
        return left.strip(), "<=", float(right.strip())
    if ">" in text:
        left, right = text.split(">", 1)
        return left.strip(), ">", float(right.strip())
    raise ValueError(f"Unsupported condition: {text}")


def rule_matches_row(rule_text: str, row: pd.Series) -> bool:
    rule_text = str(rule_text).strip()
    if rule_text.upper() == "TRUE":
        return True

    atoms = [x.strip() for x in rule_text.split("AND") if x.strip()]
    for atom in atoms:
        feature, op, threshold = parse_atomic_condition(atom)
        if feature not in row.index:
            return False
        val = to_number(row[feature])
        if np.isnan(val):
            return False
        if op == "<=" and not (val <= threshold):
            return False
        if op == ">" and not (val > threshold):
            return False
    return True


def assign_rules_to_feature_rows(feature_df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.copy()
    out["matched_leaf_id"] = np.nan
    out["matched_rule"] = None
    out["matched_predicted_class"] = np.nan

    if "sample_count" in rules_df.columns:
        rules_iter = rules_df.sort_values("sample_count", ascending=False).to_dict("records")
    else:
        rules_iter = rules_df.to_dict("records")

    matched_leaf_ids = []
    matched_rules = []
    matched_classes = []

    for _, row in out.iterrows():
        found_leaf = np.nan
        found_rule = None
        found_class = np.nan

        for rule in rules_iter:
            rule_text = str(rule.get("rule", ""))
            if rule_matches_row(rule_text, row):
                found_leaf = rule.get("leaf_id", np.nan)
                found_rule = rule_text
                found_class = rule.get("predicted_class", np.nan)
                break

        matched_leaf_ids.append(found_leaf)
        matched_rules.append(found_rule)
        matched_classes.append(found_class)

    out["matched_leaf_id"] = matched_leaf_ids
    out["matched_rule"] = matched_rules
    out["matched_predicted_class"] = matched_classes
    return out


def plot_rule_activation_timeline(
    features_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    episode_id: int,
    agent: str,
    out_path: str | Path,
) -> Path:
    df = features_df.copy()
    df["agent"] = df["agent"].astype(str)
    df = df[(df["episode_id"] == episode_id) & (df["agent"] == str(agent))].copy()
    df = df.sort_values("step_id")

    if df.empty:
        raise ValueError(f"No rows in features_df for episode_id={episode_id}, agent={agent}")

    df = assign_rules_to_feature_rows(df, rules_df)
    matched = df["matched_leaf_id"].notna().sum()

    out_path = Path(out_path)
    if matched == 0:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Для выбранной траектории не удалось сопоставить\nни одно текстовое правило с feature-строками.",
            ha="center", va="center", fontsize=13,
        )
        ax.set_title(f"Активация правил на траектории: episode={episode_id}, agent={agent}")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(df["step_id"], df["matched_leaf_id"], marker="o", label="matched_rule_leaf")
    ax.set_xlabel("Шаг")
    ax.set_ylabel("leaf_id правила")
    ax.set_title(f"Активация правил на траектории: episode={episode_id}, agent={agent}")

    labels = [ACTION_LABELS.get(maybe_int(a), str(a)) for a in df["action"]]
    for x, y, txt in zip(df["step_id"], df["matched_leaf_id"], labels):
        if not pd.isna(y):
            ax.text(x, y, txt, fontsize=8)

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================================================
# ------------------------- main --------------------------
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced two-coin visualizations for Coin Game")
    parser.add_argument("--steps-jsonl", required=True)
    parser.add_argument("--per-agent-csv", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--rules-csv", required=True)
    parser.add_argument("--rule-feature-frequency-csv", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--agent", default="0")
    parser.add_argument("--board-size", type=int, default=3)
    parser.add_argument("--top-k-rules", type=int, default=10)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    steps = load_steps_jsonl(args.steps_jsonl)
    per_agent_df = load_csv(args.per_agent_csv)
    features_df = load_csv(args.features_csv)
    rules_df = load_csv(args.rules_csv)

    # 1. Visualizations of trajectories
    plot_episode_frames_two_coins(
        steps=steps,
        episode_id=args.episode_id,
        out_path=out_dir / f"episode_{args.episode_id}_frames_two_coins.png",
        board_size=args.board_size,
    )
    plot_episode_trajectory_overlay_two_coins(
        steps=steps,
        episode_id=args.episode_id,
        out_path=out_dir / f"episode_{args.episode_id}_trajectory_overlay_two_coins.png",
        board_size=args.board_size,
    )

    # 2. Event timelines
    plot_event_timeline_two_coins(
        per_agent_df=per_agent_df,
        episode_id=args.episode_id,
        agent=args.agent,
        out_path=out_dir / f"episode_{args.episode_id}_agent_{args.agent}_event_timeline.png",
    )
    plot_reciprocity_timeline(
        per_agent_df=per_agent_df,
        episode_id=args.episode_id,
        out_path=out_dir / f"episode_{args.episode_id}_reciprocity_timeline.png",
    )

    # 3. Heatmaps and frequency maps
    plot_visit_heatmap(
        per_agent_df=per_agent_df,
        out_path=out_dir / "visit_heatmap_all.png",
        agent=None,
        board_size=args.board_size,
    )
    plot_visit_heatmap(
        per_agent_df=per_agent_df,
        out_path=out_dir / f"visit_heatmap_agent_{args.agent}.png",
        agent=args.agent,
        board_size=args.board_size,
    )
    plot_pick_heatmap(
        per_agent_df=per_agent_df,
        out_path=out_dir / f"own_pick_heatmap_agent_{args.agent}.png",
        pick_kind="own",
        agent=args.agent,
        board_size=args.board_size,
    )
    plot_pick_heatmap(
        per_agent_df=per_agent_df,
        out_path=out_dir / f"opp_pick_heatmap_agent_{args.agent}.png",
        pick_kind="opp",
        agent=args.agent,
        board_size=args.board_size,
    )
    plot_action_frequency_by_coin_relation(
        per_agent_df=per_agent_df,
        out_path=out_dir / "action_frequency_by_coin_relation.png",
    )

    # 4. Surrogate model visualizations
    plot_top_rules_table(
        rules_df=rules_df,
        out_path=out_dir / "top_rules_table.png",
        top_k=args.top_k_rules,
    )
    if args.rule_feature_frequency_csv:
        freq_df = load_csv(args.rule_feature_frequency_csv)
        plot_rule_feature_frequency(
            freq_df=freq_df,
            out_path=out_dir / "rule_feature_frequency.png",
        )

    # 5. Rule-to-trajectory alignment
    plot_rule_activation_timeline(
        features_df=features_df,
        rules_df=rules_df,
        episode_id=args.episode_id,
        agent=args.agent,
        out_path=out_dir / f"episode_{args.episode_id}_agent_{args.agent}_rule_activation.png",
    )

    print(f"Done. Visualizations saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
