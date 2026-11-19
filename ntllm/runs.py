"""Run directories and logs, so every run's config, hypothesis and results land in files."""
import json
import time
from datetime import datetime

from ntllm.paths import RUNS


class RunLog:
    """runs/<model>/<timestamp>_<name>/ with run.log (everything printed) and config.json."""

    def __init__(self, model: str, name: str, config: dict):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = RUNS / model / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        self.log(f"run dir: {self.dir}")

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        with open(self.dir / "run.log", "a") as f:
            f.write(msg + "\n")

    def header(self, hypothesis: str | None, predicted: str | None) -> None:
        """Competition Lab rule: hypothesis + predicted change go in the log before the run."""
        self.log(f"HYPOTHESIS: {hypothesis or '(none given)'}")
        self.log(f"PREDICTED:  {predicted or '(none given)'}")

    def elapsed(self) -> float:
        return time.time() - self.t0
