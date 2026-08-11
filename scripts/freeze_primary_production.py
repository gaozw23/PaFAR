"""Create the immutable PRIMARY production source/configuration snapshot and lock."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import subprocess
import zipfile

from pafar_sim.config import effective_config_checksum, load_config, project_root
from pafar_sim.io_utils import atomic_write_json, environment_manifest, file_checksum, implementation_checksum


EXCLUDED_PREFIXES = (
    ".venv/", "outputs/smoke/", "outputs/diagnostics/", "outputs/production_pilot/",
    "outputs/production/raw/", "outputs/production/calibration_maxima/", "outputs/production/learner_metrics/",
)


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    if "__pycache__" in path.parts or relative.endswith(('.pyc', '.pyo')):
        return False
    return not any(relative == prefix.rstrip("/") or relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def main() -> int:
    root = project_root(); production = root / "outputs" / "production"
    production.mkdir(parents=True, exist_ok=True)
    unexpected = [p for p in production.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if unexpected:
        raise SystemExit(f"Refusing to lock nonempty production directory: {[str(p) for p in unexpected]}")
    exp1 = load_config(root / "configs" / "exp1_primary.yaml").data
    exp2 = load_config(root / "configs" / "exp2_primary.yaml").data
    if int(exp1["master_seed"]) != int(exp2["master_seed"]):
        raise SystemExit("Primary configurations have different master seeds")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "archives" / f"PaFAR_primary_production_lock_{timestamp}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != archive and _included(path, root):
                bundle.write(path, path.relative_to(root).as_posix())
    os.chmod(archive, stat.S_IREAD)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    git_commit = git.stdout.strip() if git.returncode == 0 else None
    manifest = environment_manifest(root, effective_config_checksum(exp1), int(exp1["master_seed"]))
    lock = {
        "lock_version": 1, "timestamp_utc": timestamp, "git_commit": git_commit,
        "git_status": "tracked" if git_commit else "not_a_git_repository",
        "implementation_checksum": implementation_checksum(),
        "effective_config_checksums": {
            "exp1_primary": effective_config_checksum(exp1), "exp2_primary": effective_config_checksum(exp2),
        },
        "config_files": {
            "exp1_primary": "configs/exp1_primary.yaml", "exp2_primary": "configs/exp2_primary.yaml",
        },
        "pdf_checksums": {"PaFAR.pdf": file_checksum(root / "PaFAR.pdf"), "v40i08.pdf": file_checksum(root / "v40i08.pdf")},
        "python": manifest["python"], "platform": manifest["platform"], "packages": manifest["packages"],
        "master_seed": int(exp1["master_seed"]), "alpha_grid": exp1["alpha_grid"],
        "standardization_prevalences": exp1["standardization_prevalences"],
        "replicate_counts": {"S1": 500, "S2": 500, "S3": 500, "S4": 500, "E1": 100, "E2": 100, "E3": 50},
        "oracle_nref": int(exp1["experiment1"]["oracle_nref"]), "n_jobs": 4,
        "snapshot_path": str(archive.relative_to(root).as_posix()), "snapshot_checksum": file_checksum(archive),
        "production_started": False, "sensitivity_simulations_run": False,
    }
    atomic_write_json(lock, production / "PRODUCTION_LOCK.json")
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
