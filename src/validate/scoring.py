"""Lazy reusable adapter to the existing standalone PartiScore API."""
from pathlib import Path
import os
from partidesign.paths import ENV_MODEL_DIR


class ScoreBackend:
    def __init__(self, repo_root, config, device="cuda:0"):
        self.repo_root, self.config, self.device = Path(repo_root), config, device
        self.scorer = None

    def score(self, path, design_chains):
        if self.scorer is None:
            from ..scoring.api import PartiScorer
            checkpoint = Path(self.config["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = (Path.cwd() if ENV_MODEL_DIR in os.environ else self.repo_root) / checkpoint
            self.scorer = PartiScorer.from_checkpoint(
                checkpoint, device=self.device, weights=self.config["weights"],
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"],
                max_pair_elements=self.config.get("max_pair_elements", 2097152))
        result = self.scorer.score_files([str(path)], design_chains=[list(design_chains)],
                                         scope="design_groups")[0]
        return result.as_dict()

    def close(self):
        if self.scorer is not None:
            self.scorer.close()
            self.scorer = None
