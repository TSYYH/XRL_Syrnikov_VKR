from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover
    raise RuntimeError("CoinGameEnvModule requires JAX to be installed.") from exc

AgentId = str
ObsDict = Dict[AgentId, Any]
ActionDict = Dict[AgentId, int]
RewardDict = Dict[AgentId, float]
DoneDict = Dict[str, bool]
InfoDict = Dict[str, Any]
StepRecord = Dict[str, Any]
PolicyFn = Callable[[ObsDict, Any, Tuple[AgentId, ...]], Mapping[AgentId, int]]


@dataclass(slots=True)
class CoinGameEnvConfig:
    env_name: str = "coin_game"
    seed: int = 0
    env_kwargs: Optional[Dict[str, Any]] = None
    auto_reset: bool = False


class CoinGameEnvModule:
    """Thin wrapper around JaxMARL Coin Game with explicit two-coin semantics."""

    def __init__(self, config: Optional[CoinGameEnvConfig] = None) -> None:
        self.config = config or CoinGameEnvConfig()
        try:
            from jaxmarl import make as jaxmarl_make
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CoinGameEnvModule requires jaxmarl to be installed and importable.") from exc
        self.env = jaxmarl_make(self.config.env_name, **(self.config.env_kwargs or {}))
        self.agents: List[str] = list(self.env.agents)
        self.num_agents: int = int(self.env.num_agents)
        self._rng = jax.random.PRNGKey(int(self.config.seed))
        self.state: Any = None
        self.obs: Optional[ObsDict] = None
        self.last_info: Optional[InfoDict] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[ObsDict, Any]:
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        key = self._split_one()
        obs, state = self.env.reset(key)
        self.obs = self._to_numpy_tree(obs)
        self.state = state
        self.last_info = None
        return self.obs, state

    def step(self, actions: Mapping[AgentId, int]) -> Tuple[ObsDict, RewardDict, DoneDict, InfoDict, Any]:
        if self.state is None:
            if not self.config.auto_reset:
                raise RuntimeError("Environment must be reset() before step().")
            self.reset()

        key = self._split_one()
        jax_actions = {str(agent): jnp.asarray(int(actions[agent]), dtype=jnp.int32) for agent in self.agents}
        obs, state, rewards, dones, info = self.env.step(key, self.state, jax_actions)
        self.obs = self._to_numpy_tree(obs)
        self.state = state
        self.last_info = self._to_numpy_tree(info)
        return (
            self.obs,
            self._coerce_reward_dict(rewards),
            self._coerce_done_dict(dones),
            self.last_info,
            state,
        )

    def sample_random_actions(self) -> ActionDict:
        keys = self._split_many(self.num_agents)
        out: ActionDict = {}
        for idx, agent in enumerate(self.agents):
            out[str(agent)] = int(self.env.action_space(agent).sample(keys[idx]))
        return out

    def semantic_snapshot(self, state: Any = None, info: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        state = self.state if state is None else state
        state_dict = self._object_to_dict(state)
        info_dict = self._object_to_dict(info or {})

        red_pos = self._pick(state_dict, info_dict, "red_pos", "red_position", "agent_0_pos")
        blue_pos = self._pick(state_dict, info_dict, "blue_pos", "blue_position", "agent_1_pos")
        red_coin_pos = self._pick(state_dict, info_dict, "red_coin_pos")
        blue_coin_pos = self._pick(state_dict, info_dict, "blue_coin_pos")
        step_t = self._pick(state_dict, info_dict, "inner_t", "outer_t", "t", "step_count")

        red_coin_list = self._int_list_or_none(red_coin_pos)
        blue_coin_list = self._int_list_or_none(blue_coin_pos)

        # Legacy compatibility only. New analysis should use explicit red/blue coins.
        coin_pos = self._pick(state_dict, info_dict, "coin_pos", "coin_position")
        coin_color = self._pick(state_dict, info_dict, "coin_color", "coin_type", "coin_owner")
        if coin_pos is None and red_coin_list is not None and blue_coin_list is not None and red_coin_list == blue_coin_list:
            coin_pos = red_coin_list
            if coin_color is None:
                coin_color = "both"

        return {
            "red_pos": self._int_list_or_none(red_pos),
            "blue_pos": self._int_list_or_none(blue_pos),
            "red_coin_pos": red_coin_list,
            "blue_coin_pos": blue_coin_list,
            "coin_pos": self._int_list_or_none(coin_pos),
            "coin_color": self._scalar_or_none(coin_color),
            "t": self._scalar_or_none(step_t),
            "raw_state": self._to_numpy_tree(state_dict),
        }

    def build_step_record(
        self,
        *,
        episode_id: int,
        step_id: int,
        actions: Mapping[AgentId, int],
        rewards: Mapping[AgentId, float],
        dones: Mapping[str, bool],
        info: Optional[Mapping[str, Any]] = None,
        obs: Optional[ObsDict] = None,
        state: Any = None,
    ) -> StepRecord:
        obs = self.obs if obs is None else obs
        state = self.state if state is None else state
        semantic = self.semantic_snapshot(state=state, info=info)
        return {
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "actions": {str(k): int(v) for k, v in actions.items()},
            "rewards": {str(k): float(v) for k, v in rewards.items()},
            "dones": {str(k): bool(v) for k, v in dones.items()},
            "observations": self._to_numpy_tree(obs),
            "semantic_state": semantic,
            "raw_info": self._to_numpy_tree(info or {}),
        }

    def rollout_episode(
        self,
        policy_fn: PolicyFn,
        *,
        episode_id: int = 0,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[StepRecord]:
        obs, state = self.reset(seed=seed)
        records: List[StepRecord] = []
        step_id = 0
        while True:
            actions = dict(policy_fn(obs, state, tuple(self.agents)))
            obs, rewards, dones, info, state = self.step(actions)
            records.append(
                self.build_step_record(
                    episode_id=episode_id,
                    step_id=step_id,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    info=info,
                    obs=obs,
                    state=state,
                )
            )
            step_id += 1
            if bool(dones.get("__all__", False)):
                break
            if max_steps is not None and step_id >= int(max_steps):
                break
        return records

    def _split_one(self):
        self._rng, subkey = jax.random.split(self._rng)
        return subkey

    def _split_many(self, n: int):
        keys = jax.random.split(self._rng, n + 1)
        self._rng = keys[0]
        return tuple(keys[1:])

    def _to_numpy_tree(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._to_numpy_tree(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_numpy_tree(v) for v in value]
        if is_dataclass(value):
            return self._to_numpy_tree(asdict(value))
        if hasattr(value, "_asdict"):
            return self._to_numpy_tree(value._asdict())
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist() if np.ndim(value) > 0 else value.item()
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            arr = np.asarray(value)
            return arr.tolist() if arr.ndim > 0 else arr.item()
        return value

    def _object_to_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "_asdict"):
            return dict(value._asdict())
        if hasattr(value, "__dict__"):
            return {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return {}

    def _pick(self, state_dict: Mapping[str, Any], info_dict: Mapping[str, Any], *names: str):
        for name in names:
            if name in state_dict:
                return state_dict[name]
            if name in info_dict:
                return info_dict[name]
        return None

    def _coerce_reward_dict(self, rewards: Mapping[str, Any]) -> RewardDict:
        return {str(k): float(np.asarray(v).reshape(())) for k, v in rewards.items()}

    def _coerce_done_dict(self, dones: Mapping[str, Any]) -> DoneDict:
        return {str(k): bool(np.asarray(v).reshape(())) for k, v in dones.items()}

    def _int_list_or_none(self, value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return None
        return [int(x) for x in arr[:2]]

    def _scalar_or_none(self, value: Any) -> Any:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        if arr.size == 1:
            return arr.reshape(()).item()
        return arr.tolist()
