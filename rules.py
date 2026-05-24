from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree, export_text

DEFAULT_LEAKAGE_COLUMNS = {
    "reward",
    "rewards",
    "done",
    "dones",
    "done_all",
    "global_done",
    "episode_return",
    "episode_returns",
    "returned_episode",
    "returned_episode_returns",
    "next_reward",
    "future_reward",
    "episode_id",
    "step_id",
    "picked_any_coin",
    "picked_own_coin",
    "picked_opp_coin",
    "picked_red_coin",
    "picked_blue_coin",
    "realized_defection",
    "realized_cooperation",
    "action_label",
    "action_changed_from_prev",
}

@dataclass(slots=True)
class RuleExtractionConfig:
    target_column: str = "action"
    feature_columns: Optional[Sequence[str]] = None
    exclude_columns: Sequence[str] = field(default_factory=lambda: tuple(sorted(DEFAULT_LEAKAGE_COLUMNS)))
    dropna: bool = True
    tree_max_depth: int = 4
    tree_min_samples_leaf: int = 5
    tree_min_samples_split: int = 10
    tree_ccp_alpha: float = 0.0
    random_state: int = 0
    test_size: float = 0.25
    stratify: bool = True

@dataclass(slots=True)
class RuleExtractionResult:
    rules_df: pd.DataFrame
    tree_text: str
    feature_columns: List[str]
    target_column: str
    class_names: List[str]
    tree_depth: int
    n_leaves: int
    train_accuracy: float
    test_accuracy: float
    majority_accuracy: float
    majority_class: str
    train_rows: int
    test_rows: int
    class_distribution: Dict[str, int]
    rules_csv_path: Optional[str] = None
    tree_txt_path: Optional[str] = None
    excluded_columns: List[str] = field(default_factory=list)

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "rules_csv": self.rules_csv_path,
            "tree_txt": self.tree_txt_path,
            "num_rules": int(len(self.rules_df)),
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "tree_depth": int(self.tree_depth),
            "n_leaves": int(self.n_leaves),
            "train_accuracy": float(self.train_accuracy),
            "test_accuracy": float(self.test_accuracy),
            "majority_accuracy": float(self.majority_accuracy),
            "majority_class": str(self.majority_class),
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
            "class_names": list(self.class_names),
            "class_distribution": dict(self.class_distribution),
            "excluded_columns": list(self.excluded_columns),
        }

class CoinGameRuleExtractionModule:
    def __init__(self, config: Optional[RuleExtractionConfig] = None):
        self.config = config or RuleExtractionConfig()
        self.model: Optional[DecisionTreeClassifier] = None
        self.feature_columns_: List[str] = []
        self.class_names_: List[str] = []
        self.excluded_columns_: List[str] = []
        self._fitted_X: Optional[pd.DataFrame] = None
        self._fitted_y: Optional[pd.Series] = None
        self._leaf_ids: Optional[np.ndarray] = None

    def run(self, features_df: pd.DataFrame, output_dir: Optional[str | Path] = None, *, rules_filename: str = "coin_game_rules.csv", tree_filename: str = "coin_game_tree.txt") -> RuleExtractionResult:
        result = self.fit(features_df)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            rules_path = output_dir / rules_filename
            tree_path = output_dir / tree_filename
            result.rules_df.to_csv(rules_path, index=False)
            tree_path.write_text(result.tree_text, encoding="utf-8")
            result.rules_csv_path = str(rules_path.resolve())
            result.tree_txt_path = str(tree_path.resolve())
        return result

    def fit(self, features_df: pd.DataFrame) -> RuleExtractionResult:
        df = self._prepare_dataframe(features_df)
        X, y = self._split_xy(df)
        X_train, X_test, y_train, y_test = self._train_test_split(X, y)

        model = DecisionTreeClassifier(
            max_depth=self.config.tree_max_depth,
            min_samples_leaf=self.config.tree_min_samples_leaf,
            min_samples_split=self.config.tree_min_samples_split,
            ccp_alpha=self.config.tree_ccp_alpha,
            random_state=self.config.random_state,
        )
        model.fit(X_train, y_train)

        self.model = model
        self.feature_columns_ = list(X_train.columns)
        self.class_names_ = [str(v) for v in model.classes_]
        self._fitted_X = X_train.reset_index(drop=True)
        self._fitted_y = y_train.reset_index(drop=True)
        self._leaf_ids = model.apply(X_train)

        rules_df = self._build_rules_df()
        tree_text = export_text(model, feature_names=self.feature_columns_, show_weights=False, decimals=4)
        train_accuracy = float(model.score(X_train, y_train))
        test_accuracy = float(model.score(X_test, y_test)) if len(X_test) else train_accuracy

        majority_class = str(y_train.mode(dropna=False).iloc[0])
        majority_accuracy = float((y_test.astype(str) == majority_class).mean()) if len(y_test) else float((y_train.astype(str) == majority_class).mean())
        class_distribution = {str(k): int(v) for k, v in y.astype(str).value_counts().sort_index().items()}

        return RuleExtractionResult(
            rules_df=rules_df,
            tree_text=tree_text,
            feature_columns=list(self.feature_columns_),
            target_column=self.config.target_column,
            class_names=list(self.class_names_),
            tree_depth=int(model.get_depth()),
            n_leaves=int(model.get_n_leaves()),
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            majority_accuracy=majority_accuracy,
            majority_class=majority_class,
            train_rows=int(len(X_train)),
            test_rows=int(len(X_test)),
            class_distribution=class_distribution,
            excluded_columns=list(self.excluded_columns_),
        )

    def fit_and_export(self, features_df: pd.DataFrame, rules_csv_path: str | Path, tree_txt_path: str | Path) -> RuleExtractionResult:
        rules_csv_path = Path(rules_csv_path)
        tree_txt_path = Path(tree_txt_path)
        return self.run(features_df, output_dir=rules_csv_path.parent, rules_filename=rules_csv_path.name, tree_filename=tree_txt_path.name)

    def _prepare_dataframe(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if self.config.target_column not in features_df.columns:
            raise KeyError(f"Target column '{self.config.target_column}' not found in features_df")
        df = features_df.copy()
        if self.config.dropna:
            df = df.dropna(axis=0).reset_index(drop=True)
        if df.empty:
            raise ValueError("No rows left after preprocessing features_df")
        return df

    def _split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        target = self.config.target_column
        y = df[target].copy()
        if self.config.feature_columns is not None:
            requested = [c for c in self.config.feature_columns if c in df.columns]
            if not requested:
                raise ValueError("No configured feature_columns found in features_df")
            feature_columns = [c for c in requested if c != target and pd.api.types.is_numeric_dtype(df[c])]
            self.excluded_columns_ = []
        else:
            excluded = set(self.config.exclude_columns)
            excluded.add(target)
            self.excluded_columns_ = sorted(c for c in df.columns if c in excluded)
            feature_columns = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
        if not feature_columns:
            raise ValueError("No numeric feature columns available for rule extraction")
        return df[feature_columns].copy(), y

    def _train_test_split(self, X: pd.DataFrame, y: pd.Series):
        if not 0.0 < float(self.config.test_size) < 1.0 or len(X) < 4:
            return X.copy(), X.iloc[0:0].copy(), y.copy(), y.iloc[0:0].copy()
        stratify = y if self.config.stratify and y.nunique() > 1 and y.value_counts().min() >= 2 else None
        try:
            return train_test_split(X, y, test_size=float(self.config.test_size), random_state=int(self.config.random_state), stratify=stratify)
        except ValueError:
            return train_test_split(X, y, test_size=float(self.config.test_size), random_state=int(self.config.random_state), stratify=None)

    def _build_rules_df(self) -> pd.DataFrame:
        if self.model is None or self._fitted_X is None or self._fitted_y is None:
            raise RuntimeError("Model is not fitted")
        tree = self.model.tree_
        leaf_to_conditions = self._collect_leaf_conditions(tree)
        leaf_ids = np.asarray(self._leaf_ids)
        y = self._fitted_y.reset_index(drop=True)
        records: List[Dict[str, Any]] = []
        class_values = list(self.model.classes_)
        for leaf_id, conditions in sorted(leaf_to_conditions.items(), key=lambda kv: kv[0]):
            mask = leaf_ids == leaf_id
            y_leaf = y[mask]
            if len(y_leaf) == 0:
                continue
            counts = y_leaf.value_counts()
            predicted_class = counts.idxmax()
            purity = float(counts.max() / len(y_leaf))
            record: Dict[str, Any] = {
                "leaf_id": int(leaf_id),
                "rule": " AND ".join(conditions) if conditions else "TRUE",
                "sample_count": int(len(y_leaf)),
                "predicted_class": predicted_class,
                "empirical_accuracy": purity,
                "purity": purity,
            }
            for class_value in class_values:
                record[f"class_count_{self._safe_class_name(class_value)}"] = int(counts.get(class_value, 0))
            records.append(record)
        return pd.DataFrame(records).sort_values(["sample_count", "leaf_id"], ascending=[False, True]).reset_index(drop=True)

    def _collect_leaf_conditions(self, tree) -> Dict[int, List[str]]:
        feature_names = self.feature_columns_
        out: Dict[int, List[str]] = {}
        def walk(node_id: int, path: List[str]) -> None:
            feature_idx = tree.feature[node_id]
            if feature_idx == _tree.TREE_UNDEFINED:
                out[node_id] = list(path)
                return
            feature_name = feature_names[feature_idx]
            threshold = float(tree.threshold[node_id])
            walk(tree.children_left[node_id], path + [f"{feature_name} <= {threshold:.4f}"])
            walk(tree.children_right[node_id], path + [f"{feature_name} > {threshold:.4f}"])
        walk(0, [])
        return out

    @staticmethod
    def _safe_class_name(value: Any) -> str:
        text = str(value)
        chars = [ch if ch.isalnum() else "_" for ch in text]
        cleaned = "".join(chars).strip("_")
        return cleaned or "class"

def extract_rules(features_df: pd.DataFrame, config: Optional[RuleExtractionConfig] = None) -> RuleExtractionResult:
    return CoinGameRuleExtractionModule(config=config).fit(features_df)

def extract_rules_to_files(features_df: pd.DataFrame, rules_csv_path: str | Path, tree_txt_path: str | Path, config: Optional[RuleExtractionConfig] = None) -> RuleExtractionResult:
    module = CoinGameRuleExtractionModule(config=config)
    return module.fit_and_export(features_df, rules_csv_path, tree_txt_path)
