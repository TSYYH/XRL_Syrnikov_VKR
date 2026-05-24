from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:
    from .features import FeatureConfig, CoinGameFeatureEngineer
except ImportError:
    from features import FeatureConfig, CoinGameFeatureEngineer

JsonDict = Dict[str, Any]


@dataclass(slots=True)
class TreeSurrogateConfig:
    agents: Tuple[str, str] = ("0", "1")
    history_window: int = 5
    fill_value: float = 0.0


class TreeSurrogatePolicy:
    def __init__(
        self,
        *,
        model: Any,
        feature_columns: Sequence[str],
        agents: Sequence[str] = ("0", "1"),
        history_window: int = 5,
        fill_value: float = 0.0,
    ) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        self.agents = tuple(str(a) for a in agents)
        agent_pair = tuple(self.agents[:2]) if len(self.agents) >= 2 else tuple(self.agents)
        self.config = TreeSurrogateConfig(
            agents=agent_pair,
            history_window=max(1, int(history_window)),
            fill_value=float(fill_value),
        )
        self._fe = CoinGameFeatureEngineer(
            FeatureConfig(
                agents=agent_pair,
                include_history=True,
                history_window=self.config.history_window,
            )
        )
        self._history: Dict[str, List[JsonDict]] = {agent: [] for agent in self.agents}

    def reset_episode(self, agents: Optional[Sequence[str]] = None) -> None:
        if agents is not None:
            self.agents = tuple(str(a) for a in agents)
        self._history = {agent: [] for agent in self.agents}

    def __call__(self, obs: Any = None, state: Any = None, agents: Optional[Sequence[str]] = None, **_: Any) -> Dict[str, int]:
        agent_order = tuple(str(a) for a in (agents or self.agents))
        decision_state = self._semantic_from_state(state)
        rows = [self._build_feature_row(decision_state, agent, agent_order) for agent in agent_order]
        X = pd.DataFrame(rows)
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = self.config.fill_value
        X = X[self.feature_columns].copy()
        for col in X.columns:
            X[col] = X[col].map(self._numericize)
        preds = self.model.predict(X)
        return {agent: int(pred) for agent, pred in zip(agent_order, preds)}

    def observe_transition(
        self,
        *,
        decision_state: Mapping[str, Any],
        post_state: Mapping[str, Any],
        actions: Mapping[str, Any],
        rewards: Mapping[str, Any],
        dones: Mapping[str, Any],
        info: Optional[Mapping[str, Any]] = None,
        agents: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> None:
        agent_order = tuple(str(a) for a in (agents or self.agents))
        pickup_info = self._infer_pickups(decision_state, post_state, agent_order)
        for agent in agent_order:
            row = self._build_base_row(decision_state, agent, agent_order)
            row.update(
                {
                    "action": self._maybe_int(actions.get(agent) if actions else None),
                    "reward": self._maybe_float(rewards.get(agent) if rewards else None),
                    "done": bool(dones.get(agent, False)) if dones else False,
                    "done_all": bool(dones.get("__all__", False)) if dones else False,
                    **pickup_info.get(agent, self._empty_pick_info()),
                }
            )
            self._history.setdefault(agent, []).append(row)
            if len(self._history[agent]) > self.config.history_window:
                self._history[agent] = self._history[agent][-self.config.history_window :]

    def _build_feature_row(self, decision_state: Mapping[str, Any], agent: str, agent_order: Sequence[str]) -> JsonDict:
        row = self._build_base_row(decision_state, agent, agent_order)
        opp = str(row.get("opp_agent") or self._other_agent(agent, agent_order) or "")
        row.update(
            self._fe._build_history_features(  # type: ignore[attr-defined]
                current_row=row,
                self_history=self._history.get(agent, []),
                opp_history=self._history.get(opp, []),
            )
        )
        return row

    def _build_base_row(self, decision_state: Mapping[str, Any], agent: str, agent_order: Sequence[str]) -> JsonDict:
        opp = str(self._other_agent(agent, agent_order) or "")
        red_agent = agent_order[0] if len(agent_order) >= 1 else agent
        blue_agent = agent_order[1] if len(agent_order) >= 2 else opp

        red_pos = self._as_pair(decision_state.get("red_pos"))
        blue_pos = self._as_pair(decision_state.get("blue_pos"))
        red_coin_pos = self._as_pair(decision_state.get("red_coin_pos"))
        blue_coin_pos = self._as_pair(decision_state.get("blue_coin_pos"))

        agent_pos = red_pos if agent == red_agent else blue_pos
        opp_pos = blue_pos if opp == blue_agent else red_pos if opp == red_agent else (None, None)
        own_coin_pos = red_coin_pos if agent == red_agent else blue_coin_pos
        opp_coin_pos = blue_coin_pos if agent == red_agent else red_coin_pos

        primary_coin_pos, primary_owner, primary_relation = self._select_primary_coin(agent, agent_pos, own_coin_pos, opp_coin_pos, opp)

        agent_row, agent_col = agent_pos
        opp_row, opp_col = opp_pos
        own_row, own_col = own_coin_pos
        opp_coin_row, opp_coin_col = opp_coin_pos
        primary_row, primary_col = primary_coin_pos

        d_agent_own = self._fe._manhattan(agent_row, agent_col, own_row, own_col)  # type: ignore[attr-defined]
        d_agent_opp = self._fe._manhattan(agent_row, agent_col, opp_coin_row, opp_coin_col)  # type: ignore[attr-defined]
        d_opp_own = self._fe._manhattan(opp_row, opp_col, own_row, own_col)  # type: ignore[attr-defined]
        d_opp_opp = self._fe._manhattan(opp_row, opp_col, opp_coin_row, opp_coin_col)  # type: ignore[attr-defined]
        d_primary_agent = self._fe._manhattan(agent_row, agent_col, primary_row, primary_col)  # type: ignore[attr-defined]
        d_primary_opp = self._fe._manhattan(opp_row, opp_col, primary_row, primary_col)  # type: ignore[attr-defined]
        d_agents = self._fe._manhattan(agent_row, agent_col, opp_row, opp_col)  # type: ignore[attr-defined]

        return {
            "agent": agent,
            "opp_agent": opp,
            "agent_row": agent_row,
            "agent_col": agent_col,
            "opp_row": opp_row,
            "opp_col": opp_col,

            "red_coin_row": red_coin_pos[0],
            "red_coin_col": red_coin_pos[1],
            "blue_coin_row": blue_coin_pos[0],
            "blue_coin_col": blue_coin_pos[1],
            "own_coin_row": own_row,
            "own_coin_col": own_col,
            "opp_coin_row": opp_coin_row,
            "opp_coin_col": opp_coin_col,

            "coin_row": primary_row,
            "coin_col": primary_col,
            "coin_owner": primary_owner,
            "coin_owner_relation": primary_relation,
            "t": decision_state.get("t"),

            "agent_index": self._fe._agent_index(agent),  # type: ignore[attr-defined]
            "opp_index": self._fe._agent_index(opp),  # type: ignore[attr-defined]

            "distance_agent_to_own_coin": d_agent_own,
            "distance_agent_to_opp_coin": d_agent_opp,
            "distance_opp_to_own_coin": d_opp_own,
            "distance_opp_to_opp_coin": d_opp_opp,
            "own_coin_distance_advantage": None if d_agent_own is None or d_opp_own is None else d_opp_own - d_agent_own,
            "opp_coin_distance_advantage": None if d_agent_opp is None or d_opp_opp is None else d_opp_opp - d_agent_opp,

            "delta_row_agent_own_coin": self._fe._sub(own_row, agent_row),  # type: ignore[attr-defined]
            "delta_col_agent_own_coin": self._fe._sub(own_col, agent_col),  # type: ignore[attr-defined]
            "delta_row_agent_opp_coin": self._fe._sub(opp_coin_row, agent_row),  # type: ignore[attr-defined]
            "delta_col_agent_opp_coin": self._fe._sub(opp_coin_col, agent_col),  # type: ignore[attr-defined]

            "delta_row_opp_own_coin": self._fe._sub(own_row, opp_row),  # type: ignore[attr-defined]
            "delta_col_opp_own_coin": self._fe._sub(own_col, opp_col),  # type: ignore[attr-defined]
            "delta_row_opp_opp_coin": self._fe._sub(opp_coin_row, opp_row),  # type: ignore[attr-defined]
            "delta_col_opp_opp_coin": self._fe._sub(opp_coin_col, opp_col),  # type: ignore[attr-defined]

            "same_row_as_own_coin": self._fe._eq(agent_row, own_row),  # type: ignore[attr-defined]
            "same_col_as_own_coin": self._fe._eq(agent_col, own_col),  # type: ignore[attr-defined]
            "same_row_as_opp_coin": self._fe._eq(agent_row, opp_coin_row),  # type: ignore[attr-defined]
            "same_col_as_opp_coin": self._fe._eq(agent_col, opp_coin_col),  # type: ignore[attr-defined]

            "opp_same_row_as_own_coin": self._fe._eq(opp_row, own_row),  # type: ignore[attr-defined]
            "opp_same_col_as_own_coin": self._fe._eq(opp_col, own_col),  # type: ignore[attr-defined]
            "opp_same_row_as_opp_coin": self._fe._eq(opp_row, opp_coin_row),  # type: ignore[attr-defined]
            "opp_same_col_as_opp_coin": self._fe._eq(opp_col, opp_coin_col),  # type: ignore[attr-defined]

            "is_own_coin_contested": self._fe._eq(d_agent_own, d_opp_own),  # type: ignore[attr-defined]
            "is_opp_coin_contested": self._fe._eq(d_agent_opp, d_opp_opp),  # type: ignore[attr-defined]
            "agent_closer_to_own_coin": self._fe._lt(d_agent_own, d_opp_own),  # type: ignore[attr-defined]
            "opp_closer_to_own_coin": self._fe._lt(d_opp_own, d_agent_own),  # type: ignore[attr-defined]
            "agent_closer_to_opp_coin": self._fe._lt(d_agent_opp, d_opp_opp),  # type: ignore[attr-defined]
            "opp_closer_to_opp_coin": self._fe._lt(d_opp_opp, d_agent_opp),  # type: ignore[attr-defined]

            # legacy/primary features
            "coin_owner_is_self": primary_owner == agent,
            "coin_owner_is_opp": primary_owner == opp,
            "distance_agent_to_coin": d_primary_agent,
            "distance_opp_to_coin": d_primary_opp,
            "distance_between_agents": d_agents,
            "distance_advantage": None if d_primary_agent is None or d_primary_opp is None else d_primary_opp - d_primary_agent,
            "delta_row_agent_coin": self._fe._sub(primary_row, agent_row),  # type: ignore[attr-defined]
            "delta_col_agent_coin": self._fe._sub(primary_col, agent_col),  # type: ignore[attr-defined]
            "delta_row_opp_coin": self._fe._sub(primary_row, opp_row),  # type: ignore[attr-defined]
            "delta_col_opp_coin": self._fe._sub(primary_col, opp_col),  # type: ignore[attr-defined]
            "same_row_as_coin": self._fe._eq(agent_row, primary_row),  # type: ignore[attr-defined]
            "same_col_as_coin": self._fe._eq(agent_col, primary_col),  # type: ignore[attr-defined]
            "opp_same_row_as_coin": self._fe._eq(opp_row, primary_row),  # type: ignore[attr-defined]
            "opp_same_col_as_coin": self._fe._eq(opp_col, primary_col),  # type: ignore[attr-defined]
            "is_coin_contested": self._fe._eq(d_primary_agent, d_primary_opp),  # type: ignore[attr-defined]
            "agent_closer_to_coin": self._fe._lt(d_primary_agent, d_primary_opp),  # type: ignore[attr-defined]
            "opp_closer_to_coin": self._fe._lt(d_primary_opp, d_primary_agent),  # type: ignore[attr-defined]
        }

    def _semantic_from_state(self, state: Any) -> Dict[str, Any]:
        return {
            "red_pos": self._extract_field(state, "red_pos"),
            "blue_pos": self._extract_field(state, "blue_pos"),
            "red_coin_pos": self._extract_field(state, "red_coin_pos"),
            "blue_coin_pos": self._extract_field(state, "blue_coin_pos"),
            "t": self._maybe_int(
                self._extract_field(state, "inner_t")
                or self._extract_field(state, "t")
                or self._extract_field(state, "outer_t")
            ),
        }

    def _infer_pickups(
        self,
        decision_state: Mapping[str, Any],
        post_state: Mapping[str, Any],
        agents: Sequence[str],
    ) -> Dict[str, Dict[str, bool]]:
        out = {str(a): self._empty_pick_info() for a in agents}
        if len(agents) < 2:
            return out

        red_agent = str(agents[0])
        blue_agent = str(agents[1])

        red_coin = self._as_pair(decision_state.get("red_coin_pos"))
        blue_coin = self._as_pair(decision_state.get("blue_coin_pos"))
        post_red_pos = self._as_pair(post_state.get("red_pos"))
        post_blue_pos = self._as_pair(post_state.get("blue_pos"))

        if self._same_pos(post_red_pos, red_coin):
            out[red_agent]["picked_red_coin"] = True
        if self._same_pos(post_red_pos, blue_coin):
            out[red_agent]["picked_blue_coin"] = True
        if self._same_pos(post_blue_pos, red_coin):
            out[blue_agent]["picked_red_coin"] = True
        if self._same_pos(post_blue_pos, blue_coin):
            out[blue_agent]["picked_blue_coin"] = True

        for agent in agents:
            picked_red = out[str(agent)]["picked_red_coin"]
            picked_blue = out[str(agent)]["picked_blue_coin"]
            out[str(agent)]["picked_any_coin"] = bool(picked_red or picked_blue)
            if str(agent) == red_agent:
                out[str(agent)]["picked_own_coin"] = bool(picked_red)
                out[str(agent)]["picked_opp_coin"] = bool(picked_blue)
            else:
                out[str(agent)]["picked_own_coin"] = bool(picked_blue)
                out[str(agent)]["picked_opp_coin"] = bool(picked_red)
        return out

    def _select_primary_coin(
        self,
        agent: str,
        agent_pos: Tuple[Optional[int], Optional[int]],
        own_coin_pos: Tuple[Optional[int], Optional[int]],
        opp_coin_pos: Tuple[Optional[int], Optional[int]],
        opp: str,
    ) -> Tuple[Tuple[Optional[int], Optional[int]], Optional[str], str]:
        d_own = self._fe._manhattan(agent_pos[0], agent_pos[1], own_coin_pos[0], own_coin_pos[1])  # type: ignore[attr-defined]
        d_opp = self._fe._manhattan(agent_pos[0], agent_pos[1], opp_coin_pos[0], opp_coin_pos[1])  # type: ignore[attr-defined]
        if d_own is None and d_opp is None:
            return (None, None), None, "unknown"
        if d_opp is None or (d_own is not None and d_own <= d_opp):
            return own_coin_pos, agent, "self"
        return opp_coin_pos, opp, "opp"

    def _extract_field(self, obj: Any, name: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
            for nested_name in ("env_state", "state", "raw_state"):
                nested = obj.get(nested_name)
                if nested is not None:
                    value = self._extract_field(nested, name)
                    if value is not None:
                        return value
            return None
        if hasattr(obj, name):
            return getattr(obj, name)
        for nested_name in ("env_state", "state", "raw_state"):
            nested = getattr(obj, nested_name, None)
            if nested is not None:
                value = self._extract_field(nested, name)
                if value is not None:
                    return value
        return None

    def _other_agent(self, agent: str, agents: Sequence[str]) -> Optional[str]:
        for candidate in agents:
            if str(candidate) != str(agent):
                return str(candidate)
        return None

    def _as_pair(self, value: Any) -> Tuple[Optional[int], Optional[int]]:
        if value is None:
            return (None, None)
        if isinstance(value, Mapping):
            return (None, None)
        try:
            seq = list(value)
        except Exception:
            return (None, None)
        if len(seq) < 2:
            return (None, None)
        return (self._maybe_int(seq[0]), self._maybe_int(seq[1]))

    @staticmethod
    def _same_pos(a: Tuple[Optional[int], Optional[int]], b: Tuple[Optional[int], Optional[int]]) -> bool:
        return a != (None, None) and b != (None, None) and a == b

    @staticmethod
    def _empty_pick_info() -> Dict[str, bool]:
        return {
            "picked_red_coin": False,
            "picked_blue_coin": False,
            "picked_any_coin": False,
            "picked_own_coin": False,
            "picked_opp_coin": False,
        }

    def _maybe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _maybe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _numericize(self, value: Any) -> float:
        if value is None or value == "":
            return self.config.fill_value
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
            return self.config.fill_value
