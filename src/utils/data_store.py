from __future__ import annotations
import json
from pathlib import Path
from src.core.schemas import AnalogyResult


class AnalogyStore:
    """Lưu lịch sử analogy theo user (chuẩn bị cho Bậc 3)."""

    def __init__(self, output_dir: Path = Path("artifacts/analogies")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: AnalogyResult, user_id: str = "anonymous") -> str:
        file_path = self.output_dir / f"{user_id}.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")
        return str(file_path)

    def load_history(self, user_id: str) -> list[dict]:
        file_path = self.output_dir / f"{user_id}.jsonl"
        if not file_path.exists():
            return []
        with open(file_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]