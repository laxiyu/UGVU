"""UGVU configuration system 鈥?YAML-based config with dataclass binding."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env without overwriting env vars."""
    env_path = Path(path)
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _expand_env_vars(value):
    """Recursively expand ${VAR_NAME} placeholders in loaded YAML values."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single black-box generator model."""
    name: str = "doubao"
    api_endpoint: str = ""
    api_key: str = ""
    model_version: str = "latest"
    api_mode: str = "chat"            # For qwen: "chat" | "generation"
    timeout_sec: int = 60
    max_retries: int = 3
    image_size: tuple = (1024, 1024)
    temperature: float = 1.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRegistry:
    """Registry of all available models."""
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelRegistry":
        known_fields = set(ModelConfig.__dataclass_fields__.keys()) - {"name", "extra_params"}
        models = {}
        for name, raw_cfg in d.items():
            cfg = dict(raw_cfg or {})
            extra_params = dict(cfg.pop("extra_params", {}) or {})
            known = {}
            for key, value in cfg.items():
                if key in known_fields:
                    known[key] = value
                else:
                    extra_params[key] = value
            models[name] = ModelConfig(name=name, extra_params=extra_params, **known)
        return cls(models=models)

    def get(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise KeyError(f"Model '{name}' not in registry. Available: {list(self.models.keys())}")
        return self.models[name]


# ---------------------------------------------------------------------------
# Sampling configs
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    """Configuration for K-shot sampling."""
    k_samples: int = 5
    parallel: bool = True
    max_concurrent: int = 4
    seed: int = 42
    output_dir: str = "outputs/samples"
    adaptive: bool = False
    min_samples: int = 3
    uncertainty_threshold: float = 0.15
    check_interval: int = 1


# ---------------------------------------------------------------------------
# Consensus configs
# ---------------------------------------------------------------------------

@dataclass
class ConsensusConfig:
    """Configuration for consensus fusion."""
    method: str = "ugcf"                # "majority", "ugcf", "cmcf"
    uncertainty_type: str = "entropy"   # "entropy", "variance"
    weight_temperature: float = 1.0     # Temperature for softmax over uncertainty weights
    cross_model_models: List[str] = field(default_factory=lambda: ["doubao", "qwen"])
    iou_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Calibration configs
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Configuration for reliability calibration analysis."""
    num_bins: int = 15
    strategy: str = "uniform"  # "uniform" or "quantile"
    metrics: List[str] = field(default_factory=lambda: ["ece", "mce", "spearman"])


# ---------------------------------------------------------------------------
# Robustness configs
# ---------------------------------------------------------------------------

@dataclass
class RobustnessConfig:
    """Configuration for GVU-Robust benchmark."""
    num_prompt_variants: int = 50
    generation_repeats: int = 20
    models_to_compare: List[str] = field(default_factory=lambda: ["doubao", "qwen"])
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])


# ---------------------------------------------------------------------------
# Task / dataset configs
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Configuration for a specific task and dataset."""
    task: str = "semantic_segmentation"   # semantic_segmentation, referring_segmentation, depth_estimation
    dataset: str = "cityscapes"
    data_root: str = "data/"
    num_classes: int = 19
    ignore_index: int = 255
    batch_size: int = 1
    num_workers: int = 4
    max_samples: Optional[int] = None    # Limit for debugging; None = full dataset


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class UGVUConfig:
    """Top-level UGVU configuration."""
    experiment_name: str = "ugvu_default"
    seed: int = 42
    output_dir: str = "outputs/"

    models: ModelRegistry = field(default_factory=ModelRegistry)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UGVUConfig":
        """Load configuration from a YAML file."""
        _load_dotenv(Path(path).parent / ".env")
        _load_dotenv(Path.cwd() / ".env")
        with open(path, "r", encoding="utf-8") as f:
            raw = _expand_env_vars(yaml.safe_load(f) or {})

        models_raw = raw.get("models", {})
        model_registry = ModelRegistry.from_dict(models_raw)

        return cls(
            experiment_name=raw.get("experiment_name", "ugvu_default"),
            seed=raw.get("seed", 42),
            output_dir=raw.get("output_dir", "outputs/"),
            models=model_registry,
            sampling=SamplingConfig(**raw.get("sampling", {})),
            consensus=ConsensusConfig(**raw.get("consensus", {})),
            calibration=CalibrationConfig(**raw.get("calibration", {})),
            robustness=RobustnessConfig(**raw.get("robustness", {})),
            task=TaskConfig(**raw.get("task", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        d = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "models": {
                name: {
                    k: v for k, v in cfg.__dict__.items()
                    if k != "name" and not k.startswith("_")
                }
                for name, cfg in self.models.models.items()
            },
            "sampling": {k: v for k, v in self.sampling.__dict__.items() if not k.startswith("_")},
            "consensus": {k: v for k, v in self.consensus.__dict__.items() if not k.startswith("_")},
            "calibration": {k: v for k, v in self.calibration.__dict__.items() if not k.startswith("_")},
            "robustness": {k: v for k, v in self.robustness.__dict__.items() if not k.startswith("_")},
            "task": {k: v for k, v in self.task.__dict__.items() if not k.startswith("_")},
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
