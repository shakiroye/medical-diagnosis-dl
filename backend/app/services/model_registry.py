from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedModelInfo:
    name: str
    path: Path
    available: bool


class ModelRegistry:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir

    def resolve(self, model_name: str) -> LoadedModelInfo:
        candidates = [
            self.model_dir / f"{model_name}.keras",
            self.model_dir / model_name / "saved_model.pb",
        ]
        for candidate in candidates:
            if candidate.exists():
                return LoadedModelInfo(name=model_name, path=candidate, available=True)
        return LoadedModelInfo(name=model_name, path=candidates[0], available=False)
