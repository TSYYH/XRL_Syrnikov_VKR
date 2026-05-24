from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import pandas as pd

try:
    from .environment import CoinGameEnvConfig, CoinGameEnvModule
    from .training import MAPPOTrainingConfig, JaxMARLMAPPOTrainer
    from .coin_game_logging import CoinGameLogger
    from .features import FeatureConfig, CoinGameFeatureEngineer
    from .rules import CoinGameRuleExtractionModule, RuleExtractionConfig
    from .visualization import CoinGameVisualizer, VisualizationConfig
    from .surrogate_policy import TreeSurrogatePolicy
except ImportError:
    from environment import CoinGameEnvConfig, CoinGameEnvModule
    from training import MAPPOTrainingConfig, JaxMARLMAPPOTrainer
    from coin_game_logging import CoinGameLogger
    from features import FeatureConfig, CoinGameFeatureEngineer
    from rules import CoinGameRuleExtractionModule, RuleExtractionConfig
    from visualization import CoinGameVisualizer, VisualizationConfig
    from surrogate_policy import TreeSurrogatePolicy


PolicyFn = Callable[[Dict[str, Any], Any, Sequence[str]], Mapping[str, int]]


class CoinGamePipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_dir = Path(args.project_dir).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.paths = {
            "logs_dir": self.project_dir / "logs",
            "features_dir": self.project_dir / "features",
            "rules_dir": self.project_dir / "rules",
            "viz_dir": self.project_dir / "visualizations",
            "artifacts_dir": self.project_dir / "artifacts",
        }
        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)

        self.env_module = CoinGameEnvModule(
            CoinGameEnvConfig(
                env_name=args.env_name,
                seed=args.seed,
                env_kwargs=self._parse_json_dict(args.env_kwargs),
            )
        )
        self.logger = CoinGameLogger(self.env_module.agents, output_dir=self.paths["logs_dir"])
        self.feature_engineer = CoinGameFeatureEngineer(
            FeatureConfig(
                agents=tuple(str(a) for a in self.env_module.agents[:2]),
                include_history=not bool(args.disable_history_features),
                history_window=int(args.history_window),
            )
        )
        self.rule_extractor = CoinGameRuleExtractionModule(
            RuleExtractionConfig(
                target_column=args.rule_target_column,
                tree_max_depth=int(args.tree_max_depth),
                tree_min_samples_leaf=int(args.tree_min_samples_leaf),
                tree_min_samples_split=int(args.tree_min_samples_split),
                tree_ccp_alpha=float(args.tree_ccp_alpha),
                random_state=int(args.rule_random_state),
                test_size=float(args.rule_test_size),
                stratify=not bool(args.disable_rule_stratify),
            )
        )
        self.visualizer = CoinGameVisualizer(VisualizationConfig(output_dir=self.paths["viz_dir"]))

    def run(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {
            "project_dir": str(self.project_dir),
            "env_name": self.args.env_name,
            "seed": int(self.args.seed),
            "num_episodes": int(self.args.num_episodes),
            "max_steps_per_episode": int(self.args.max_steps_per_episode),
            "collection_policy": self.args.collection_policy,
            "paths": {k: str(v) for k, v in self.paths.items()},
            "training": None,
            "exports": {},
            "visualizations": {},
            "rules": {},
            "feature_config": {
                "include_history": not bool(self.args.disable_history_features),
                "history_window": int(self.args.history_window),
            },
        }

        if self.args.run_training:
            training_result = self.run_training()
            manifest["training"] = self._to_jsonable(training_result)

        policy_fn = self.make_policy_fn()
        self.collect_episodes(policy_fn)
        original_episode_summaries = self.logger.summarize_all()
        original_mean_return = self._mean_episode_return(original_episode_summaries)

        export_paths = self.logger.save_all(prefix=self.args.export_prefix)
        manifest["exports"].update(export_paths)

        feature_rows = self.feature_engineer.build_from_csv(export_paths["per_agent_csv"])
        features_csv = self.feature_engineer.save_csv(
            feature_rows,
            self.paths["features_dir"] / f"{self.args.export_prefix}_features.csv",
        )
        manifest["exports"]["features_csv"] = str(features_csv.resolve())
        manifest["dataset_rows"] = len(feature_rows)
        surrogate_eval = manifest.setdefault("surrogate_evaluation", {})
        surrogate_eval["original_mean_return"] = original_mean_return

        if feature_rows:
            features_df = pd.DataFrame(feature_rows)
            try:
                rule_result = self.rule_extractor.run(
                    features_df,
                    output_dir=self.paths["rules_dir"],
                    rules_filename=f"{self.args.export_prefix}_rules.csv",
                    tree_filename=f"{self.args.export_prefix}_tree.txt",
                )
                manifest["rules"].update(rule_result.to_manifest_dict())
                manifest["rules"]["config"] = {
                    "tree_max_depth": int(self.args.tree_max_depth),
                    "tree_min_samples_leaf": int(self.args.tree_min_samples_leaf),
                    "tree_min_samples_split": int(self.args.tree_min_samples_split),
                    "tree_ccp_alpha": float(self.args.tree_ccp_alpha),
                    "rule_test_size": float(self.args.rule_test_size),
                    "rule_random_state": int(self.args.rule_random_state),
                    "stratify": not bool(self.args.disable_rule_stratify),
                }

                if self.rule_extractor.model is not None:
                    tree_png = self.visualizer.plot_decision_tree(
                        self.rule_extractor.model,
                        rule_result.feature_columns,
                        filepath=self.paths["viz_dir"] / f"{self.args.export_prefix}_decision_tree.png",
                    )
                    manifest["visualizations"]["decision_tree_png"] = str(tree_png.resolve())
            except Exception as exc:
                manifest["rules"]["error"] = str(exc)

            if self.args.compute_return_gap and self.rule_extractor.model is not None:
                try:
                    surrogate_result = self.evaluate_surrogate_policy(
                        model=self.rule_extractor.model,
                        feature_columns=rule_result.feature_columns,
                        original_mean_return=original_mean_return,
                    )
                    manifest.setdefault("surrogate_evaluation", {}).update(surrogate_result)
                except Exception as exc:
                    manifest.setdefault("surrogate_evaluation", {})["error"] = str(exc)

            try:
                reward_png = self.visualizer.plot_episode_rewards(
                    feature_rows,
                    episode_id=int(self.args.preview_episode_id),
                    filepath=self.paths["viz_dir"] / f"{self.args.export_prefix}_episode_{self.args.preview_episode_id}_rewards.png",
                )
                manifest["visualizations"]["episode_rewards_png"] = str(reward_png.resolve())
            except Exception as exc:
                manifest["visualizations"]["episode_rewards_png_error"] = str(exc)

            try:
                heatmap_png = self.visualizer.plot_behavior_heatmap(
                    feature_rows,
                    filepath=self.paths["viz_dir"] / f"{self.args.export_prefix}_defection_heatmap.png",
                )
                manifest["visualizations"]["defection_heatmap_png"] = str(heatmap_png.resolve())
            except Exception as exc:
                manifest["visualizations"]["defection_heatmap_png_error"] = str(exc)

            try:
                board_png = self._render_preview_board()
                if board_png is not None:
                    manifest["visualizations"]["preview_board_png"] = str(board_png.resolve())
            except Exception as exc:
                manifest["visualizations"]["preview_board_png_error"] = str(exc)

        manifest_path = self.project_dir / "pipeline_manifest.json"
        manifest["manifest_path"] = str(manifest_path.resolve())
        manifest_path.write_text(json.dumps(self._to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def make_policy_fn(self) -> PolicyFn:
        if self.args.collection_policy == "random":
            return self._make_random_policy_fn()
        if self.args.collection_policy == "python":
            if not self.args.policy_module_path:
                raise ValueError("--policy-module-path is required when --collection-policy=python")
            return self._load_policy_from_python_file(self.args.policy_module_path, checkpoint_path=self.args.checkpoint_path)
        raise ValueError(f"Unsupported collection policy: {self.args.collection_policy}")

    def _make_random_policy_fn(self) -> PolicyFn:
        def policy_fn(obs: Dict[str, Any], state: Any, agents: Sequence[str]) -> Dict[str, int]:
            return self.env_module.sample_random_actions()
        return policy_fn

    def _load_policy_from_python_file(self, path: str | Path, *, checkpoint_path: Optional[str]) -> PolicyFn:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"policy module not found: {path}")

        module_name = f"coin_game_policy_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        candidate_names = [
            self.args.policy_factory_name,
            "make_policy_fn",
            "build_policy_fn",
            "make_policy",
            "build_policy",
            "policy_fn",
            "policy",
        ]
        candidate_names = [name for name in candidate_names if name]

        for name in candidate_names:
            if hasattr(module, name):
                obj = getattr(module, name)
                if callable(obj):
                    return self._instantiate_policy_callable(obj, checkpoint_path=checkpoint_path)

        raise AttributeError(f"No supported policy factory found in {path}. Tried: {candidate_names}")

    def _instantiate_policy_callable(self, obj: Callable[..., Any], *, checkpoint_path: Optional[str]) -> PolicyFn:
        common_kwargs = {
            "checkpoint_path": checkpoint_path,
            "agents": tuple(self.env_module.agents),
            "env": self.env_module,
            "env_module": self.env_module,
            "env_name": self.args.env_name,
        }

        for kwargs in (
            common_kwargs,
            {"checkpoint_path": checkpoint_path, "agents": tuple(self.env_module.agents)},
            {"checkpoint_path": checkpoint_path},
            {},
        ):
            try:
                produced = obj(**{k: v for k, v in kwargs.items() if v is not None})
                if callable(produced):
                    return self._wrap_policy_fn(produced)
            except TypeError:
                continue

        return self._wrap_policy_fn(obj)

    def _wrap_policy_fn(self, policy_callable: Callable[..., Any]) -> PolicyFn:
        def policy_fn(obs: Dict[str, Any], state: Any, agents: Sequence[str]) -> Dict[str, int]:
            for kwargs in (
                {"obs": obs, "state": state, "agents": tuple(agents)},
                {"obs_dict": obs, "state": state, "agents": tuple(agents)},
                {"obs": obs, "agents": tuple(agents)},
                {"obs_dict": obs, "agents": tuple(agents)},
                {},
            ):
                try:
                    out = policy_callable(**kwargs) if kwargs else policy_callable(obs, state, tuple(agents))
                    return {str(k): int(v) for k, v in dict(out).items()}
                except TypeError:
                    continue
            out = policy_callable(obs, state, tuple(agents))
            return {str(k): int(v) for k, v in dict(out).items()}

        for method_name in ("reset_episode", "observe_transition"):
            if hasattr(policy_callable, method_name):
                setattr(policy_fn, method_name, getattr(policy_callable, method_name))
        return policy_fn

    def collect_episodes(
        self,
        policy_fn: PolicyFn,
        *,
        env_module: Optional[CoinGameEnvModule] = None,
        logger: Optional[CoinGameLogger] = None,
        num_episodes: Optional[int] = None,
        max_steps_per_episode: Optional[int] = None,
        seed_offset: Optional[int] = None,
    ) -> CoinGameLogger:
        env_module = env_module or self.env_module
        logger = logger or self.logger
        num_episodes = int(self.args.num_episodes if num_episodes is None else num_episodes)
        max_steps_per_episode = int(self.args.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode)
        seed_offset = int(self.args.seed if seed_offset is None else seed_offset)

        for episode_id in range(num_episodes):
            if hasattr(policy_fn, "reset_episode"):
                policy_fn.reset_episode(tuple(env_module.agents))

            obs, state = env_module.reset(seed=seed_offset + episode_id)
            step_id = 0
            while True:
                decision_state = env_module.semantic_snapshot(state=state)
                actions = dict(policy_fn(obs, state, tuple(env_module.agents)))
                next_obs, rewards, dones, info, next_state = env_module.step(actions)
                post_state = env_module.semantic_snapshot(state=next_state, info=info)

                step_record = {
                    "episode_id": int(episode_id),
                    "step_id": int(step_id),
                    "actions": {str(k): int(v) for k, v in actions.items()},
                    "rewards": {str(k): float(v) for k, v in rewards.items()},
                    "dones": {str(k): bool(v) for k, v in dones.items()},
                    "observations": obs,
                    "semantic_state": post_state,
                    "decision_state": decision_state,
                    "raw_info": info or {},
                }
                logger.log_step(step_record)

                if hasattr(policy_fn, "observe_transition"):
                    policy_fn.observe_transition(
                        decision_state=decision_state,
                        post_state=post_state,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        info=info,
                        agents=tuple(env_module.agents),
                        episode_id=episode_id,
                        step_id=step_id,
                    )

                obs, state = next_obs, next_state
                step_id += 1
                if bool(dones.get("__all__", False)):
                    break
                if max_steps_per_episode is not None and step_id >= int(max_steps_per_episode):
                    break

        return logger

    def evaluate_surrogate_policy(
        self,
        *,
        model: Any,
        feature_columns: Sequence[str],
        original_mean_return: float,
    ) -> Dict[str, Any]:
        surrogate_policy = TreeSurrogatePolicy(
            model=model,
            feature_columns=feature_columns,
            agents=tuple(self.env_module.agents),
            history_window=int(self.args.history_window),
        )
        surrogate_env = CoinGameEnvModule(
            CoinGameEnvConfig(
                env_name=self.args.env_name,
                seed=int(self.args.seed),
                env_kwargs=self._parse_json_dict(self.args.env_kwargs),
            )
        )
        surrogate_logger = CoinGameLogger(
            surrogate_env.agents,
            output_dir=self.paths["logs_dir"] / "surrogate",
        )
        surrogate_num_episodes = int(self.args.surrogate_eval_episodes or self.args.num_episodes)
        surrogate_max_steps = int(self.args.surrogate_eval_max_steps or self.args.max_steps_per_episode)

        self.collect_episodes(
            surrogate_policy,
            env_module=surrogate_env,
            logger=surrogate_logger,
            num_episodes=surrogate_num_episodes,
            max_steps_per_episode=surrogate_max_steps,
            seed_offset=int(self.args.seed),
        )

        export_paths = surrogate_logger.save_all(prefix=self.args.surrogate_export_prefix)
        summaries = surrogate_logger.summarize_all()
        surrogate_mean_return = self._mean_episode_return(summaries)
        return_gap = float(original_mean_return - surrogate_mean_return)
        return {
            "enabled": True,
            "num_episodes": surrogate_num_episodes,
            "max_steps_per_episode": surrogate_max_steps,
            "surrogate_mean_return": surrogate_mean_return,
            "return_gap": return_gap,
            "abs_return_gap": abs(return_gap),
            "exports": export_paths,
        }

    def _mean_episode_return(self, summaries: Sequence[Any]) -> float:
        if not summaries:
            return 0.0
        values = [float(getattr(summary, "mean_reward", 0.0)) for summary in summaries]
        return float(sum(values) / len(values))

    def run_training(self) -> Any:
        if not self.args.jaxmarl_repo_root:
            raise ValueError("--jaxmarl-repo-root is required when --run-training is used")

        trainer = JaxMARLMAPPOTrainer(
            MAPPOTrainingConfig(
                repo_root=self.args.jaxmarl_repo_root,
                env_name=self.args.env_name,
                seed=int(self.args.seed),
                num_seeds=int(self.args.train_num_seeds),
                save_path=str((self.project_dir / "training_run").resolve()),
                project=self.args.train_project,
                wandb_mode=self.args.train_wandb_mode,
                extra_overrides=self._parse_overrides(self.args.train_override),
                env=dict(os.environ),
            )
        )
        return trainer.train(capture_logs=True)

    def _render_preview_board(self) -> Optional[Path]:
        first = next(iter(self.logger.iter_step_records()), None)
        if first is None:
            return None
        return self.visualizer.render_board_state(first, filepath=self.paths["viz_dir"] / f"{self.args.export_prefix}_preview_board.png")

    def _parse_json_dict(self, value: str) -> Dict[str, Any]:
        value = (value or "{}").strip()
        if not value:
            return {}
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("env_kwargs must decode to a JSON object")
        return parsed

    def _parse_overrides(self, items: Optional[Sequence[str]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for item in items or []:
            if "=" not in item:
                raise ValueError(f"Invalid --train-override entry: {item!r}. Expected KEY=VALUE")
            key, raw_value = item.split("=", 1)
            result[key] = self._smart_parse(raw_value)
        return result

    def _smart_parse(self, value: str) -> Any:
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower == "null":
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        try:
            return json.loads(value)
        except Exception:
            return value

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return value


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean end-to-end Coin Game pipeline")
    p.add_argument("--project-dir", default="./coin_game_project")
    p.add_argument("--env-name", default="coin_game")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-kwargs", default="{}", help="JSON dict passed into jaxmarl.make(...)")

    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps-per-episode", type=int, default=50)
    p.add_argument("--collection-policy", choices=["random", "python"], default="random")
    p.add_argument("--policy-module-path", default=None)
    p.add_argument("--policy-factory-name", default=None)
    p.add_argument("--checkpoint-path", default=None)

    p.add_argument("--run-training", action="store_true")
    p.add_argument("--jaxmarl-repo-root", default=None)
    p.add_argument("--train-num-seeds", type=int, default=1)
    p.add_argument("--train-project", default="coin_game_mappo")
    p.add_argument("--train-wandb-mode", default="disabled")
    p.add_argument("--train-override", action="append", default=[])

    p.add_argument("--export-prefix", default="coin_game")
    p.add_argument("--preview-episode-id", type=int, default=0)

    p.add_argument("--history-window", type=int, default=5)
    p.add_argument("--disable-history-features", action="store_true")

    p.add_argument("--rule-target-column", default="action")
    p.add_argument("--tree-max-depth", type=int, default=4)
    p.add_argument("--tree-min-samples-leaf", type=int, default=5)
    p.add_argument("--tree-min-samples-split", type=int, default=10)
    p.add_argument("--tree-ccp-alpha", type=float, default=0.0)
    p.add_argument("--rule-test-size", type=float, default=0.25)
    p.add_argument("--rule-random-state", type=int, default=0)
    p.add_argument("--disable-rule-stratify", action="store_true")

    p.add_argument("--compute-return-gap", action="store_true")
    p.add_argument("--surrogate-eval-episodes", type=int, default=None)
    p.add_argument("--surrogate-eval-max-steps", type=int, default=None)
    p.add_argument("--surrogate-export-prefix", default="coin_game_surrogate")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    pipeline = CoinGamePipeline(args)
    manifest = pipeline.run()
    print(json.dumps(pipeline._to_jsonable(manifest), ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
