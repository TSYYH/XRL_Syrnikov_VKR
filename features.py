from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

JsonDict = Dict[str, Any]


@dataclass(slots=True)
class FeatureConfig:
    agents: Tuple[str, str] = ("0", "1")
    include_history: bool = True
    history_window: int = 5


class CoinGameFeatureEngineer:
    """Builds an interpretable state-action table from logger outputs.

    The fixed version preserves both coins explicitly:
    - own coin from the current agent's perspective
    - opponent coin from the current agent's perspective

    Legacy single-coin features are still emitted for compatibility and refer
    to the *primary* coin already chosen in the logger (nearest coin from the
    agent's perspective, own coin on ties).
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self.agents = tuple(self.config.agents)
        self.history_window = max(1, int(self.config.history_window))

    def build_from_rows(self, rows: Iterable[Mapping[str, Any]]) -> List[JsonDict]:
        base_rows = [self._normalize_row(r) for r in rows]
        base_rows.sort(key=lambda r: (int(r["episode_id"]), int(r["step_id"]), str(r["agent"])))

        out: List[JsonDict] = []
        history_by_agent: Dict[Tuple[int, str], List[JsonDict]] = {}

        for row in base_rows:
            episode_id = int(row["episode_id"])
            agent = str(row["agent"])
            opp = str(row.get("opp_agent") or self._other_agent(agent) or "")

            agent_row = self._as_int(row.get("agent_row"))
            agent_col = self._as_int(row.get("agent_col"))
            opp_row = self._as_int(row.get("opp_row"))
            opp_col = self._as_int(row.get("opp_col"))

            own_coin_row = self._as_int(row.get("own_coin_row"))
            own_coin_col = self._as_int(row.get("own_coin_col"))
            opp_coin_row = self._as_int(row.get("opp_coin_row"))
            opp_coin_col = self._as_int(row.get("opp_coin_col"))

            coin_row = self._as_int(row.get("coin_row"))
            coin_col = self._as_int(row.get("coin_col"))
            coin_owner = row.get("coin_owner")
            coin_owner_relation = row.get("coin_owner_relation")

            d_agent_own = self._manhattan(agent_row, agent_col, own_coin_row, own_coin_col)
            d_agent_opp = self._manhattan(agent_row, agent_col, opp_coin_row, opp_coin_col)
            d_opp_own = self._manhattan(opp_row, opp_col, own_coin_row, own_coin_col)
            d_opp_opp = self._manhattan(opp_row, opp_col, opp_coin_row, opp_coin_col)
            d_agents = self._manhattan(agent_row, agent_col, opp_row, opp_col)

            d_agent_primary = self._manhattan(agent_row, agent_col, coin_row, coin_col)
            d_opp_primary = self._manhattan(opp_row, opp_col, coin_row, coin_col)

            feature_row = dict(row)
            feature_row.update(
                {
                    "agent_index": self._agent_index(agent),
                    "opp_index": self._agent_index(opp),

                    # Explicit own/opp coin semantics
                    "distance_agent_to_own_coin": d_agent_own,
                    "distance_agent_to_opp_coin": d_agent_opp,
                    "distance_opp_to_own_coin": d_opp_own,
                    "distance_opp_to_opp_coin": d_opp_opp,
                    "own_coin_distance_advantage": None if d_agent_own is None or d_opp_own is None else d_opp_own - d_agent_own,
                    "opp_coin_distance_advantage": None if d_agent_opp is None or d_opp_opp is None else d_opp_opp - d_agent_opp,

                    "delta_row_agent_own_coin": self._sub(own_coin_row, agent_row),
                    "delta_col_agent_own_coin": self._sub(own_coin_col, agent_col),
                    "delta_row_agent_opp_coin": self._sub(opp_coin_row, agent_row),
                    "delta_col_agent_opp_coin": self._sub(opp_coin_col, agent_col),

                    "delta_row_opp_own_coin": self._sub(own_coin_row, opp_row),
                    "delta_col_opp_own_coin": self._sub(own_coin_col, opp_col),
                    "delta_row_opp_opp_coin": self._sub(opp_coin_row, opp_row),
                    "delta_col_opp_opp_coin": self._sub(opp_coin_col, opp_col),

                    "same_row_as_own_coin": self._eq(agent_row, own_coin_row),
                    "same_col_as_own_coin": self._eq(agent_col, own_coin_col),
                    "same_row_as_opp_coin": self._eq(agent_row, opp_coin_row),
                    "same_col_as_opp_coin": self._eq(agent_col, opp_coin_col),

                    "opp_same_row_as_own_coin": self._eq(opp_row, own_coin_row),
                    "opp_same_col_as_own_coin": self._eq(opp_col, own_coin_col),
                    "opp_same_row_as_opp_coin": self._eq(opp_row, opp_coin_row),
                    "opp_same_col_as_opp_coin": self._eq(opp_col, opp_coin_col),

                    "is_own_coin_contested": self._eq(d_agent_own, d_opp_own),
                    "is_opp_coin_contested": self._eq(d_agent_opp, d_opp_opp),
                    "agent_closer_to_own_coin": self._lt(d_agent_own, d_opp_own),
                    "opp_closer_to_own_coin": self._lt(d_opp_own, d_agent_own),
                    "agent_closer_to_opp_coin": self._lt(d_agent_opp, d_opp_opp),
                    "opp_closer_to_opp_coin": self._lt(d_opp_opp, d_agent_opp),

                    # Primary/legacy features retained for compatibility
                    "coin_owner_is_self": coin_owner == agent,
                    "coin_owner_is_opp": coin_owner == opp,
                    "distance_agent_to_coin": d_agent_primary,
                    "distance_opp_to_coin": d_opp_primary,
                    "distance_between_agents": d_agents,
                    "distance_advantage": None if d_agent_primary is None or d_opp_primary is None else d_opp_primary - d_agent_primary,
                    "delta_row_agent_coin": self._sub(coin_row, agent_row),
                    "delta_col_agent_coin": self._sub(coin_col, agent_col),
                    "delta_row_opp_coin": self._sub(coin_row, opp_row),
                    "delta_col_opp_coin": self._sub(coin_col, opp_col),
                    "same_row_as_coin": self._eq(agent_row, coin_row),
                    "same_col_as_coin": self._eq(agent_col, coin_col),
                    "opp_same_row_as_coin": self._eq(opp_row, coin_row),
                    "opp_same_col_as_coin": self._eq(opp_col, coin_col),
                    "is_coin_contested": self._eq(d_agent_primary, d_opp_primary),
                    "agent_closer_to_coin": self._lt(d_agent_primary, d_opp_primary),
                    "opp_closer_to_coin": self._lt(d_opp_primary, d_agent_primary),

                    # Outcome labels
                    "picked_any_coin": self._as_bool(row.get("picked_any_coin")),
                    "picked_own_coin": self._as_bool(row.get("picked_own_coin")),
                    "picked_opp_coin": self._as_bool(row.get("picked_opp_coin")),
                    "picked_red_coin": self._as_bool(row.get("picked_red_coin")),
                    "picked_blue_coin": self._as_bool(row.get("picked_blue_coin")),
                    "realized_defection": self._as_bool(row.get("picked_opp_coin")),
                    "realized_cooperation": self._as_bool(row.get("picked_own_coin")),
                    "action_label": self._action_label(row.get("action")),
                    "coin_owner_relation": (
                        coin_owner_relation if coin_owner_relation is not None else
                        ("self" if coin_owner == agent else "opp" if coin_owner == opp else "unknown")
                    ),
                }
            )

            if self.config.include_history:
                feature_row.update(
                    self._build_history_features(
                        current_row=feature_row,
                        self_history=history_by_agent.get((episode_id, agent), []),
                        opp_history=history_by_agent.get((episode_id, opp), []),
                    )
                )

            out.append(feature_row)
            history_by_agent.setdefault((episode_id, agent), []).append(feature_row)

        return out

    def build_from_csv(self, filepath: str | Path) -> List[JsonDict]:
        with Path(filepath).open("r", newline="", encoding="utf-8") as fh:
            return self.build_from_rows(csv.DictReader(fh))

    def save_csv(self, rows: Iterable[Mapping[str, Any]], filepath: str | Path) -> Path:
        rows = [dict(r) for r in rows]
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: self._csv_safe(row.get(k)) for k in fieldnames})
        return path

    def _build_history_features(
        self,
        *,
        current_row: Mapping[str, Any],
        self_history: Sequence[JsonDict],
        opp_history: Sequence[JsonDict],
    ) -> JsonDict:
        own_hist = list(self_history[-self.history_window:])
        opp_hist = list(opp_history[-self.history_window:])
        prev_self = own_hist[-1] if own_hist else None
        prev_opp = opp_hist[-1] if opp_hist else None

        return {
            "prev_action": self._get_numeric(prev_self, "action"),
            "prev_reward": self._get_numeric(prev_self, "reward"),
            "prev_picked_any_coin": self._get_bool(prev_self, "picked_any_coin"),
            "prev_picked_own_coin": self._get_bool(prev_self, "picked_own_coin"),
            "prev_picked_opp_coin": self._get_bool(prev_self, "picked_opp_coin"),
            "prev_picked_red_coin": self._get_bool(prev_self, "picked_red_coin"),
            "prev_picked_blue_coin": self._get_bool(prev_self, "picked_blue_coin"),

            "prev_distance_agent_to_coin": self._get_numeric(prev_self, "distance_agent_to_coin"),
            "prev_distance_advantage": self._get_numeric(prev_self, "distance_advantage"),
            "prev_distance_agent_to_own_coin": self._get_numeric(prev_self, "distance_agent_to_own_coin"),
            "prev_distance_agent_to_opp_coin": self._get_numeric(prev_self, "distance_agent_to_opp_coin"),
            "prev_own_coin_distance_advantage": self._get_numeric(prev_self, "own_coin_distance_advantage"),
            "prev_opp_coin_distance_advantage": self._get_numeric(prev_self, "opp_coin_distance_advantage"),

            "prev_opp_action": self._get_numeric(prev_opp, "action"),
            "prev_opp_reward": self._get_numeric(prev_opp, "reward"),
            "prev_opp_picked_any_coin": self._get_bool(prev_opp, "picked_any_coin"),
            "prev_opp_picked_own_coin": self._get_bool(prev_opp, "picked_own_coin"),
            "prev_opp_picked_opp_coin": self._get_bool(prev_opp, "picked_opp_coin"),
            "prev_opp_picked_red_coin": self._get_bool(prev_opp, "picked_red_coin"),
            "prev_opp_picked_blue_coin": self._get_bool(prev_opp, "picked_blue_coin"),

            "recent_reward_sum": self._sum_numeric(own_hist, "reward"),
            "recent_reward_mean": self._mean_numeric(own_hist, "reward"),
            "recent_defection_count": self._sum_bool(own_hist, "picked_opp_coin"),
            "recent_cooperation_count": self._sum_bool(own_hist, "picked_own_coin"),
            "recent_coin_pick_count": self._sum_bool(own_hist, "picked_any_coin"),
            "recent_red_coin_pick_count": self._sum_bool(own_hist, "picked_red_coin"),
            "recent_blue_coin_pick_count": self._sum_bool(own_hist, "picked_blue_coin"),

            "recent_opp_defection_count": self._sum_bool(opp_hist, "picked_opp_coin"),
            "recent_opp_cooperation_count": self._sum_bool(opp_hist, "picked_own_coin"),
            "recent_opp_coin_pick_count": self._sum_bool(opp_hist, "picked_any_coin"),
            "recent_opp_red_coin_pick_count": self._sum_bool(opp_hist, "picked_red_coin"),
            "recent_opp_blue_coin_pick_count": self._sum_bool(opp_hist, "picked_blue_coin"),
            "recent_opp_reward_sum": self._sum_numeric(opp_hist, "reward"),
            "recent_opp_reward_mean": self._mean_numeric(opp_hist, "reward"),

            "action_changed_from_prev": (
                prev_self is not None
                and self._get_numeric(prev_self, "action") is not None
                and current_row.get("action") is not None
                and self._get_numeric(prev_self, "action") != self._as_int(current_row.get("action"))
            ),
        }

    def _normalize_row(self, row: Mapping[str, Any]) -> JsonDict:
        return {str(k): self._parse_scalar(v) for k, v in dict(row).items()}

    def _parse_scalar(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped in {"True", "False"}:
            return stripped == "True"
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except Exception:
                return stripped
        try:
            if "." in stripped:
                return float(stripped)
            return int(stripped)
        except Exception:
            return stripped

    def _agent_index(self, agent: str) -> Optional[int]:
        try:
            return self.agents.index(agent)
        except ValueError:
            return None

    def _other_agent(self, agent: str) -> Optional[str]:
        for candidate in self.agents:
            if candidate != agent:
                return candidate
        return None

    def _as_int(self, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        return int(value)

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)

    def _manhattan(self, r1: Optional[int], c1: Optional[int], r2: Optional[int], c2: Optional[int]) -> Optional[int]:
        if None in (r1, c1, r2, c2):
            return None
        return abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))

    def _sub(self, a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None or b is None:
            return None
        return int(a) - int(b)

    def _eq(self, a: Any, b: Any) -> bool:
        return a is not None and b is not None and a == b

    def _lt(self, a: Optional[int], b: Optional[int]) -> bool:
        return a is not None and b is not None and a < b

    def _action_label(self, action: Any) -> str:
        mapping = {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}
        try:
            return mapping.get(int(action), f"action_{int(action)}")
        except Exception:
            return "unknown"

    def _csv_safe(self, value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return value

    def _get_numeric(self, row: Optional[Mapping[str, Any]], key: str) -> Optional[float]:
        if not row:
            return None
        value = row.get(key)
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return float(int(value))
        try:
            return float(value)
        except Exception:
            return None

    def _get_bool(self, row: Optional[Mapping[str, Any]], key: str) -> bool:
        if not row:
            return False
        return self._as_bool(row.get(key))

    def _sum_numeric(self, rows: Sequence[Mapping[str, Any]], key: str) -> float:
        values = [self._get_numeric(row, key) for row in rows]
        return float(sum(v for v in values if v is not None))

    def _mean_numeric(self, rows: Sequence[Mapping[str, Any]], key: str) -> float:
        values = [self._get_numeric(row, key) for row in rows]
        values = [v for v in values if v is not None]
        return float(sum(values) / len(values)) if values else 0.0

    def _sum_bool(self, rows: Sequence[Mapping[str, Any]], key: str) -> int:
        return int(sum(1 for row in rows if self._get_bool(row, key)))
