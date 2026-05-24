from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from safetensors import safe_open


def _infer_sep(keys: Iterable[str]) -> str:
    keys = list(keys)
    for sep in ("/", ".", ","):
        if any(sep in k for k in keys):
            return sep
    return "/"


def _unflatten(flat: Mapping[str, Any]) -> Dict[str, Any]:
    sep = _infer_sep(flat.keys())
    root: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = [p for p in str(key).split(sep) if p]
        cur = root
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return root


def _load_safetensors_nested(path: str | Path) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    with safe_open(str(path), framework="flax") as f:
        for key in f.keys():
            flat[key] = f.get_tensor(key)
    return _unflatten(flat)


def _extract_actor_params(tree: Mapping[str, Any]) -> Dict[str, Any]:
    if "params" in tree and isinstance(tree["params"], Mapping):
        inner = tree["params"]
        if "Dense_0" in inner and "Dense_2" in inner:
            return dict(inner)

    if "Dense_0" in tree and "Dense_2" in tree:
        return dict(tree)

    for value in tree.values():
        if isinstance(value, Mapping):
            try:
                return _extract_actor_params(value)
            except KeyError:
                pass

    raise KeyError("Could not locate actor parameters with Dense_0/Dense_2 inside checkpoint tree")


def _load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        for nested_name in ("env_state", "state"):
            if nested_name in obj:
                value = _get_field(obj[nested_name], name, None)
                if value is not None:
                    return value
        return default

    if hasattr(obj, name):
        return getattr(obj, name)

    for nested_name in ("env_state", "state"):
        nested_obj = getattr(obj, nested_name, None)
        if nested_obj is not None:
            value = _get_field(nested_obj, name, None)
            if value is not None:
                return value

    return default


def _to_python_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    return int(arr[0])


def _normalize_agents(
    agents: Optional[Iterable[Any]],
    obs: Any = None,
) -> Tuple[str, ...]:
    if agents is not None:
        return tuple(str(agent) for agent in agents)

    if isinstance(obs, Mapping):
        keys = [
            str(k)
            for k in obs.keys()
            if str(k) not in {"__all__", "world_state", "observation", "obs"}
        ]
        if keys:
            return tuple(sorted(keys))

    return ("0", "1")


def _is_numeric_like(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, int, float, np.number)):
        return True
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_numeric_like(v) for v in value) if value else False
    return False


def _flatten_obs_value(value: Any) -> Optional[jnp.ndarray]:
    if value is None:
        return None

    if isinstance(value, Mapping):
        for preferred_key in ("observation", "obs"):
            if preferred_key in value:
                flattened = _flatten_obs_value(value[preferred_key])
                if flattened is not None:
                    return flattened

        parts = []
        for key in sorted(value.keys()):
            if key in {"world_state", "__all__"}:
                continue
            flattened = _flatten_obs_value(value[key])
            if flattened is not None:
                parts.append(flattened)

        if not parts:
            return None
        return jnp.concatenate(parts, axis=0)

    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            flattened = _flatten_obs_value(item)
            if flattened is not None:
                parts.append(flattened)
        if not parts:
            return None
        return jnp.concatenate(parts, axis=0)

    if _is_numeric_like(value):
        return jnp.asarray(value, dtype=jnp.float32).reshape(-1)

    return None


def _build_abs_obs_from_state(state: Any, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    red_pos = jnp.asarray(_get_field(state, "red_pos"), dtype=jnp.int32).reshape(-1)[:2]
    blue_pos = jnp.asarray(_get_field(state, "blue_pos"), dtype=jnp.int32).reshape(-1)[:2]
    red_coin_pos = jnp.asarray(_get_field(state, "red_coin_pos"), dtype=jnp.int32).reshape(-1)[:2]
    blue_coin_pos = jnp.asarray(_get_field(state, "blue_coin_pos"), dtype=jnp.int32).reshape(-1)[:2]

    obs1 = jnp.zeros((3, 3, 4), dtype=jnp.float32)
    obs1 = obs1.at[red_pos[0], red_pos[1], 0].set(1.0)
    obs1 = obs1.at[blue_pos[0], blue_pos[1], 1].set(1.0)
    obs1 = obs1.at[red_coin_pos[0], red_coin_pos[1], 2].set(1.0)
    obs1 = obs1.at[blue_coin_pos[0], blue_coin_pos[1], 3].set(1.0)

    obs2 = jnp.stack(
        [obs1[:, :, 1], obs1[:, :, 0], obs1[:, :, 3], obs1[:, :, 2]],
        axis=-1,
    )

    if len(agents) == 1:
        return {agents[0]: obs1.reshape(-1)}

    return {
        agents[0]: obs1.reshape(-1),
        agents[1]: obs2.reshape(-1),
    }


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(x)


def _relu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.relu(x)


class CoinGamePolicy:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        greedy: bool = True,
    ) -> None:
        raw_tree = _load_safetensors_nested(checkpoint_path)
        self.params = _extract_actor_params(raw_tree)
        self.config = _load_yaml_config(config_path)

        self.obs_dim = int(jnp.asarray(self.params["Dense_0"]["kernel"]).shape[0])
        self.hidden_size = int(jnp.asarray(self.params["Dense_0"]["kernel"]).shape[1])
        self.action_dim = int(jnp.asarray(self.params["Dense_2"]["kernel"]).shape[-1])

        self.greedy = bool(greedy)
        self.hidden: Optional[jnp.ndarray] = None
        self._prev_inner_t: Optional[int] = None

    def _dense(self, x: jnp.ndarray, layer_name: str) -> jnp.ndarray:
        layer = self.params[layer_name]
        kernel = jnp.asarray(layer["kernel"], dtype=jnp.float32)
        bias = jnp.asarray(layer["bias"], dtype=jnp.float32)
        return x @ kernel + bias

    def _gru_step(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        cell = self.params["ScannedRNN_0"]["GRUCell_1"]

        ir = jnp.asarray(cell["ir"]["kernel"], dtype=jnp.float32)
        ir_b = jnp.asarray(cell["ir"]["bias"], dtype=jnp.float32)
        iz = jnp.asarray(cell["iz"]["kernel"], dtype=jnp.float32)
        iz_b = jnp.asarray(cell["iz"]["bias"], dtype=jnp.float32)
        inn = jnp.asarray(cell["in"]["kernel"], dtype=jnp.float32)
        inn_b = jnp.asarray(cell["in"]["bias"], dtype=jnp.float32)

        hr = jnp.asarray(cell["hr"]["kernel"], dtype=jnp.float32)
        hz = jnp.asarray(cell["hz"]["kernel"], dtype=jnp.float32)
        hn = jnp.asarray(cell["hn"]["kernel"], dtype=jnp.float32)
        hn_b = jnp.asarray(cell["hn"]["bias"], dtype=jnp.float32)

        r = _sigmoid(x @ ir + ir_b + h @ hr)
        z = _sigmoid(x @ iz + iz_b + h @ hz)
        n = jnp.tanh(x @ inn + inn_b + r * (h @ hn + hn_b))
        return (1.0 - z) * n + z * h

    def reset(self) -> None:
        self.hidden = None
        self._prev_inner_t = None

    def _reset_hidden_if_needed(self, batch_size: int, state: Any) -> None:
        inner_t = _get_field(state, "inner_t", None)
        inner_t_int = None if inner_t is None else _to_python_int(inner_t)

        if self.hidden is None or int(self.hidden.shape[0]) != int(batch_size):
            self.hidden = jnp.zeros((batch_size, self.hidden_size), dtype=jnp.float32)

        if inner_t_int == 0 and self._prev_inner_t not in (None, 0):
            self.hidden = jnp.zeros((batch_size, self.hidden_size), dtype=jnp.float32)

        self._prev_inner_t = inner_t_int

    def _prepare_obs(
        self,
        obs: Any,
        agents: Sequence[str],
        state: Any = None,
    ) -> jnp.ndarray:
        state_obs = None
        if state is not None:
            try:
                state_obs = _build_abs_obs_from_state(state, agents)
            except Exception:
                state_obs = None

        rows = []
        obs_map = obs if isinstance(obs, Mapping) else {}

        for agent in agents:
            candidate = None
            if isinstance(obs_map, Mapping) and agent in obs_map:
                candidate = _flatten_obs_value(obs_map[agent])

            if candidate is None and state_obs is not None and agent in state_obs:
                candidate = state_obs[agent]

            if candidate is None:
                raise ValueError(
                    f"Could not build observation for agent={agent}. "
                    f"obs keys={list(obs_map.keys()) if isinstance(obs_map, Mapping) else type(obs)}"
                )

            rows.append(jnp.asarray(candidate, dtype=jnp.float32).reshape(-1))

        obs_batch = jnp.stack(rows, axis=0)

        if obs_batch.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Observation dimension mismatch: got {obs_batch.shape[-1]}, "
                f"expected {self.obs_dim}"
            )

        return obs_batch

    def __call__(
        self,
        obs: Any,
        state: Any = None,
        agents: Optional[Iterable[Any]] = None,
        **_: Any,
    ) -> Dict[str, int]:
        agent_order = _normalize_agents(agents, obs)
        obs_batch = self._prepare_obs(obs, agent_order, state=state)
        self._reset_hidden_if_needed(obs_batch.shape[0], state)

        embedding = _relu(self._dense(obs_batch, "Dense_0"))
        self.hidden = self._gru_step(embedding, self.hidden)
        actor_hidden = _relu(self._dense(self.hidden, "Dense_1"))
        logits = self._dense(actor_hidden, "Dense_2")

        if self.greedy:
            actions = jnp.argmax(logits, axis=-1)
        else:
            actions = jax.random.categorical(jax.random.PRNGKey(0), logits, axis=-1)

        return {agent: int(actions[i]) for i, agent in enumerate(agent_order)}


def make_policy_fn(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    greedy: bool = True,
    **_: Any,
):
    policy = CoinGamePolicy(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        greedy=greedy,
    )

    def fn(obs=None, state=None, agents=None, **inner_kwargs):
        if obs is None and "obs" in inner_kwargs:
            obs = inner_kwargs.pop("obs")
        if state is None and "state" in inner_kwargs:
            state = inner_kwargs.pop("state")
        if agents is None and "agents" in inner_kwargs:
            agents = inner_kwargs.pop("agents")
        return policy(obs, state=state, agents=agents, **inner_kwargs)

    fn.reset = policy.reset  # type: ignore[attr-defined]
    fn.policy = policy       # type: ignore[attr-defined]
    return fn


build_policy = make_policy_fn
make_policy = make_policy_fn
load_policy = make_policy_fn
create_policy = make_policy_fn
POLICY_ENTRYPOINT = make_policy_fn
