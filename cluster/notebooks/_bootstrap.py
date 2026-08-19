# Databricks notebook source
# MAGIC %md
# MAGIC # Shared bootstrap
# MAGIC Do not run this notebook on its own. The other notebooks in this folder
# MAGIC pull it in with `%run ./_bootstrap`. It works out where the repo lives
# MAGIC instead of asking you to paste a path, because a mistyped path is the
# MAGIC single most common way this setup fails.

# COMMAND ----------

import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this notebook until a directory containing src/skewbench appears.

    Databricks sets the working directory to the notebook's own directory in a
    Git folder, so the repo root is always an ancestor. Searching for it beats
    hard-coding /Workspace/Users/<email>/..., which breaks the moment the
    account email or the folder name differs by one character.
    """
    here = Path(os.getcwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "skewbench" / "runner.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not find the repo root above {here}. Are you running this "
        "notebook from inside the spark-skew-bench Git folder?"
    )


REPO = _find_repo_root()
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# The one thing you may have to change. Set this to your Unity Catalog volume.
VOLUME = "/Volumes/skewbench_ws/default/skewbench"

DATA_ROOT = f"{VOLUME}/data"
RESULTS = f"{VOLUME}/results"
os.makedirs(RESULTS, exist_ok=True)

print(f"repo      {REPO}")
print(f"volume    {VOLUME}")
print(f"data      {DATA_ROOT}")
print(f"results   {RESULTS}")
