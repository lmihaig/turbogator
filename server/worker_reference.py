#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path


def log_progress(log_path, message):
    ts = datetime.now().strftime("%H:%M:%S")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{ts}] {message}\n")


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: worker_reference.py <job_id> <user> <description>", file=sys.stderr
        )
        return 1

    job_id, user, desc = sys.argv[1], sys.argv[2], sys.argv[3]
    job_dir = Path(f"/opt/aos/workspaces/{job_id}")
    artifact_dir = Path("/opt/aos/artifacts")
    workspace_tar = job_dir / "workspace.tar.gz"
    run_log = job_dir / "run.log"

    try:
        run_log.write_text("", encoding="utf-8")
        log_progress(
            run_log,
            f"Reference worker started for job_id={job_id}, user={user}, desc={desc}",
        )
        log_progress(run_log, "Extracting workspace archive...")

        with tarfile.open(workspace_tar, "r:gz") as tf:
            tf.extractall(path=job_dir)

        log_progress(run_log, "Running reference benchmark...")

        env = os.environ.copy()
        env["PATH"] = "/root/.local/bin:" + env.get("PATH", "")
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["BLIS_NUM_THREADS"] = "1"
        uv_bin = shutil.which("uv", path=env["PATH"]) or "uv"

        cmd = [
            "taskset",
            "-c",
            "2",
            uv_bin,
            "run",
            "--project",
            "/opt/aos/api",
            "reference/benchmark_pytorch.py",
        ]
        with run_log.open("a", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(cmd)}\n")
            result = subprocess.run(
                cmd,
                cwd=str(job_dir),
                stdout=log_file,
                stderr=log_file,
                text=True,
                env=env,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"reference benchmark failed with exit code {result.returncode}"
            )

        metrics_src = job_dir / "results/reference/metrics.json"
        payload = json.loads(metrics_src.read_text(encoding="utf-8"))

        metrics_out = job_dir / "metrics.json"
        metrics_out.write_text(json.dumps(payload), encoding="utf-8")
        log_progress(run_log, "Collected reference metrics.")
        log_progress(run_log, "Reference job completed successfully.")

        payload_dir = job_dir / "artifact_payload"
        payload_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_out, payload_dir / "metrics.json")
        shutil.copy2(run_log, payload_dir / "run.log")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_tar = artifact_dir / f"{job_id}.tar.gz"
        with tarfile.open(artifact_tar, "w:gz") as tf:
            tf.add(payload_dir / "metrics.json", arcname="metrics.json")
            tf.add(payload_dir / "run.log", arcname="run.log")
        return 0
    except Exception as exc:
        with run_log.open("a", encoding="utf-8") as log_file:
            log_file.write(f"ERROR: {exc}\n")

        metrics_out = job_dir / "metrics.json"
        metrics_out.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")

        payload_dir = job_dir / "artifact_payload"
        payload_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_out, payload_dir / "metrics.json")
        if run_log.exists():
            shutil.copy2(run_log, payload_dir / "run.log")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_tar = artifact_dir / f"{job_id}.tar.gz"
        with tarfile.open(artifact_tar, "w:gz") as tf:
            tf.add(payload_dir / "metrics.json", arcname="metrics.json")
            if (payload_dir / "run.log").exists():
                tf.add(payload_dir / "run.log", arcname="run.log")
        return 1
    finally:
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
