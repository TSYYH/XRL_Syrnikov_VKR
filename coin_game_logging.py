from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

JsonDict = Dict[str, Any]


@dataclass(slots=True)
class EpisodeSummary:
    episode_id: int
    steps: int
    terminated: bool
    total_reward_by_agent: Dict[str, float]
    own_coin_picks_by_agent: Dict[str, int]
    opp_coin_picks_by_agent: Dict[str, int]
    mean_reward: float


@dataclass(slots=True)
class _EpisodeBuffer:
    episode_id: int
    steps: List[JsonDict] = field(default_factory=list)
    terminated: bool = False
    total_reward_by_agent: Dict[str, float] = field(default_factory=dict)
    own_coin_picks_by_agent: Dict[str, int] = field(default_factory=dict)
    opp_coin_picks_by_agent: Dict[str, int] = field(default_factory=dict)


class CoinGameLogger:
    """Stores raw step logs plus per-agent exports with correct two-coin semantics."""

    def __init__(self, agents: Sequence[str], output_dir: str | Path = "./coin_game_logs") -> None:
        self.agents: Tuple[str, ...] = tuple(str(a) for a in agents)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episodes: Dict[int, _EpisodeBuffer] = {}
        self._steps: List[JsonDict] = []

    def log_step(self, step_record: Mapping[str, Any]) -> None:
        record = self._to_builtin(step_record)
        episode_id = int(record["episode_id"])
        buf = self._episodes.get(episode_id)
        if buf is None:
            buf = _EpisodeBuffer(
                episode_id=episode_id,
                total_reward_by_agent={agent: 0.0 for agent in self.agents},
                own_coin_picks_by_agent={agent: 0 for agent in self.agents},
                opp_coin_picks_by_agent={agent: 0 for agent in self.agents},
            )
            self._episodes[episode_id] = buf

        buf.steps.append(record)
        self._steps.append(record)

        rewards = dict(record.get("rewards", {}))
        for agent in self.agents:
            buf.total_reward_by_agent[agent] += float(rewards.get(agent, 0.0))

        pickup_info = self._infer_pickups(record)
        for agent, info in pickup_info.items():
            if info["picked_own_coin"]:
                buf.own_coin_picks_by_agent[agent] += 1
            if info["picked_opp_coin"]:
                buf.opp_coin_picks_by_agent[agent] += 1

        buf.terminated = bool(record.get("dones", {}).get("__all__", False))

    def log_episode(self, step_records: Iterable[Mapping[str, Any]]) -> None:
        for record in step_records:
            self.log_step(record)

    def iter_step_records(self) -> Iterable[JsonDict]:
        yield from self._steps

    def iter_per_agent_rows(self) -> Iterable[JsonDict]:
        for record in self._steps:
            decision = dict(record.get("decision_state") or record.get("semantic_state", {}))
            outcome = dict(record.get("semantic_state", {}))
            pickup_info = self._infer_pickups(record)

            red_pos = self._pair(decision.get("red_pos"))
            blue_pos = self._pair(decision.get("blue_pos"))
            red_coin_pos = self._pair(decision.get("red_coin_pos"))
            blue_coin_pos = self._pair(decision.get("blue_coin_pos"))

            post_red_pos = self._pair(outcome.get("red_pos"))
            post_blue_pos = self._pair(outcome.get("blue_pos"))
            post_red_coin_pos = self._pair(outcome.get("red_coin_pos"))
            post_blue_coin_pos = self._pair(outcome.get("blue_coin_pos"))

            for agent in self.agents:
                opp = self._other_agent(agent)
                agent_pos = red_pos if self._is_red_agent(agent) else blue_pos
                opp_pos = blue_pos if self._is_red_agent(agent) else red_pos

                own_coin_pos = red_coin_pos if self._is_red_agent(agent) else blue_coin_pos
                opp_coin_pos = blue_coin_pos if self._is_red_agent(agent) else red_coin_pos

                primary_coin_pos, primary_owner, primary_relation = self._select_primary_coin(
                    agent=agent,
                    agent_pos=agent_pos,
                    own_coin_pos=own_coin_pos,
                    opp_coin_pos=opp_coin_pos,
                )

                picks = pickup_info.get(agent, self._empty_pick_info())

                yield {
                    "episode_id": int(record["episode_id"]),
                    "step_id": int(record["step_id"]),
                    "agent": agent,
                    "opp_agent": opp,
                    "action": record.get("actions", {}).get(agent),
                    "reward": record.get("rewards", {}).get(agent),
                    "done": bool(record.get("dones", {}).get(agent, False)),
                    "done_all": bool(record.get("dones", {}).get("__all__", False)),
                    "agent_row": self._safe_get(agent_pos, 0),
                    "agent_col": self._safe_get(agent_pos, 1),
                    "opp_row": self._safe_get(opp_pos, 0),
                    "opp_col": self._safe_get(opp_pos, 1),
                    # explicit two-coin state
                    "red_coin_row": self._safe_get(red_coin_pos, 0),
                    "red_coin_col": self._safe_get(red_coin_pos, 1),
                    "blue_coin_row": self._safe_get(blue_coin_pos, 0),
                    "blue_coin_col": self._safe_get(blue_coin_pos, 1),
                    "own_coin_row": self._safe_get(own_coin_pos, 0),
                    "own_coin_col": self._safe_get(own_coin_pos, 1),
                    "opp_coin_row": self._safe_get(opp_coin_pos, 0),
                    "opp_coin_col": self._safe_get(opp_coin_pos, 1),
                    # legacy single-coin compatibility: nearest coin from the agent perspective
                    "coin_row": self._safe_get(primary_coin_pos, 0),
                    "coin_col": self._safe_get(primary_coin_pos, 1),
                    "coin_owner": primary_owner,
                    "coin_owner_relation": primary_relation,
                    # pickup events
                    "picked_any_coin": picks["picked_any_coin"],
                    "picked_own_coin": picks["picked_own_coin"],
                    "picked_opp_coin": picks["picked_opp_coin"],
                    "picked_red_coin": picks["picked_red_coin"],
                    "picked_blue_coin": picks["picked_blue_coin"],
                    "t": decision.get("t"),
                    "post_red_pos": post_red_pos,
                    "post_blue_pos": post_blue_pos,
                    "post_red_coin_pos": post_red_coin_pos,
                    "post_blue_coin_pos": post_blue_coin_pos,
                }

    def summarize_episode(self, episode_id: int) -> EpisodeSummary:
        buf = self._episodes[int(episode_id)]
        mean_reward = 0.0
        if buf.steps:
            totals = list(buf.total_reward_by_agent.values())
            mean_reward = float(sum(totals) / len(totals))
        return EpisodeSummary(
            episode_id=buf.episode_id,
            steps=len(buf.steps),
            terminated=buf.terminated,
            total_reward_by_agent=dict(buf.total_reward_by_agent),
            own_coin_picks_by_agent=dict(buf.own_coin_picks_by_agent),
            opp_coin_picks_by_agent=dict(buf.opp_coin_picks_by_agent),
            mean_reward=mean_reward,
        )

    def summarize_all(self) -> List[EpisodeSummary]:
        return [self.summarize_episode(episode_id) for episode_id in sorted(self._episodes)]

    def save_steps_jsonl(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in self._steps:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def save_per_agent_csv(self, filepath: str | Path) -> Path:
        return self._write_csv(filepath, list(self.iter_per_agent_rows()))

    def save_episode_summary_csv(self, filepath: str | Path) -> Path:
        rows: List[JsonDict] = []
        for summary in self.summarize_all():
            rows.append(
                {
                    "episode_id": summary.episode_id,
                    "steps": summary.steps,
                    "terminated": summary.terminated,
                    "mean_reward": summary.mean_reward,
                    **{f"reward_{k}": v for k, v in summary.total_reward_by_agent.items()},
                    **{f"own_coin_picks_{k}": v for k, v in summary.own_coin_picks_by_agent.items()},
                    **{f"opp_coin_picks_{k}": v for k, v in summary.opp_coin_picks_by_agent.items()},
                }
            )
        return self._write_csv(filepath, rows)

    def save_all(self, prefix: str = "coin_game") -> Dict[str, str]:
        steps_path = self.save_steps_jsonl(self.output_dir / f"{prefix}_steps.jsonl")
        per_agent_path = self.save_per_agent_csv(self.output_dir / f"{prefix}_per_agent.csv")
        episode_path = self.save_episode_summary_csv(self.output_dir / f"{prefix}_episodes.csv")
        return {
            "steps_jsonl": str(steps_path.resolve()),
            "per_agent_csv": str(per_agent_path.resolve()),
            "episode_csv": str(episode_path.resolve()),
        }

    # ----------------------- internals -----------------------

    def _infer_pickups(self, record: Mapping[str, Any]) -> Dict[str, Dict[str, bool]]:
        decision = dict(record.get("decision_state") or record.get("semantic_state", {}))
        outcome = dict(record.get("semantic_state", {}))

        red_coin = self._pair(decision.get("red_coin_pos"))
        blue_coin = self._pair(decision.get("blue_coin_pos"))
        post_red_pos = self._pair(outcome.get("red_pos"))
        post_blue_pos = self._pair(outcome.get("blue_pos"))

        info = {agent: self._empty_pick_info() for agent in self.agents}
        if len(self.agents) < 2:
            return info

        red_agent = self.agents[0]
        blue_agent = self.agents[1]

        if self._same_pos(post_red_pos, red_coin):
            info[red_agent]["picked_red_coin"] = True
        if self._same_pos(post_red_pos, blue_coin):
            info[red_agent]["picked_blue_coin"] = True
        if self._same_pos(post_blue_pos, red_coin):
            info[blue_agent]["picked_red_coin"] = True
        if self._same_pos(post_blue_pos, blue_coin):
            info[blue_agent]["picked_blue_coin"] = True

        for agent in self.agents:
            picked_red = info[agent]["picked_red_coin"]
            picked_blue = info[agent]["picked_blue_coin"]
            info[agent]["picked_any_coin"] = bool(picked_red or picked_blue)
            if self._is_red_agent(agent):
                info[agent]["picked_own_coin"] = bool(picked_red)
                info[agent]["picked_opp_coin"] = bool(picked_blue)
            else:
                info[agent]["picked_own_coin"] = bool(picked_blue)
                info[agent]["picked_opp_coin"] = bool(picked_red)
        return info

    def _select_primary_coin(
        self,
        *,
        agent: str,
        agent_pos: Tuple[Optional[int], Optional[int]],
        own_coin_pos: Tuple[Optional[int], Optional[int]],
        opp_coin_pos: Tuple[Optional[int], Optional[int]],
    ) -> Tuple[Tuple[Optional[int], Optional[int]], Optional[str], str]:
        d_own = self._manhattan(agent_pos, own_coin_pos)
        d_opp = self._manhattan(agent_pos, opp_coin_pos)
        opp_agent = self._other_agent(agent)

        if d_own is None and d_opp is None:
            return (None, None), None, "unknown"
        if d_opp is None or (d_own is not None and d_own <= d_opp):
            return own_coin_pos, agent, "self"
        return opp_coin_pos, opp_agent, "opp"

    def _is_red_agent(self, agent: str) -> bool:
        return bool(self.agents) and str(agent) == str(self.agents[0])

    def _other_agent(self, agent: str) -> Optional[str]:
        for candidate in self.agents:
            if candidate != agent:
                return candidate
        return None

    @staticmethod
    def _pair(value: Any) -> Tuple[Optional[int], Optional[int]]:
        if value is None:
            return (None, None)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return (None, None)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return (int(value[0]) if value[0] is not None else None, int(value[1]) if value[1] is not None else None)
            except Exception:
                return (None, None)
        return (None, None)

    @staticmethod
    def _same_pos(a: Tuple[Optional[int], Optional[int]], b: Tuple[Optional[int], Optional[int]]) -> bool:
        return a != (None, None) and b != (None, None) and a == b

    @staticmethod
    def _manhattan(a: Tuple[Optional[int], Optional[int]], b: Tuple[Optional[int], Optional[int]]) -> Optional[int]:
        if a == (None, None) or b == (None, None):
            return None
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    @staticmethod
    def _safe_get(value: Any, idx: int) -> Any:
        if isinstance(value, (list, tuple)) and len(value) > idx:
            return value[idx]
        return None

    @staticmethod
    def _empty_pick_info() -> Dict[str, bool]:
        return {
            "picked_red_coin": False,
            "picked_blue_coin": False,
            "picked_any_coin": False,
            "picked_own_coin": False,
            "picked_opp_coin": False,
        }

    def _write_csv(self, filepath: str | Path, rows: List[JsonDict]) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: self._csv_safe(row.get(k)) for k in fieldnames})
        return path

    def _csv_safe(self, value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return value

    def _to_builtin(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_builtin(v) for v in value]
        return value
