#!/usr/bin/env python3
import csv
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

METRICS_TEMP_NAME = "metrics_temp.json"
LLVM_MCA_TIMEOUT_S = 300


def run_logged(
    cmd,
    log_path,
    cwd=None,
    progress_log=None,
    label=None,
):
    command_label = label or " ".join(cmd)
    start = datetime.now()
    if progress_log is not None:
        log_progress(progress_log, f"START: {command_label}")

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n")
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=log_file,
            stderr=log_file,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if progress_log is not None:
                log_progress(
                    progress_log,
                    f"FAIL: {command_label} (exit={result.returncode})",
                )
            raise RuntimeError(
                f"command failed with exit code {result.returncode}: {' '.join(cmd)}"
            )

    elapsed = (datetime.now() - start).total_seconds()
    if progress_log is not None:
        log_progress(progress_log, f"DONE: {command_label} ({elapsed:.1f}s)")


def _perf_event_list(app_config):
    return ",".join(str(e) for e in app_config.PERF_EVENTS)


def _llvm_mca_flags(app_config):
    return [str(flag) for flag in app_config.LLVM_MCA_FLAGS]


def _load_metrics_temp(project_dir):
    metrics_path = project_dir / METRICS_TEMP_NAME
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics_path.unlink()

    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid {METRICS_TEMP_NAME} payload")
    return payload


def _parse_perf_counters(stderr_text):
    counters = {}
    reader = csv.reader(stderr_text.splitlines())
    for row in reader:
        if len(row) < 3 or not row[2]:
            continue

        key = row[2].strip().strip("'\"").replace("-", "_")

        value = row[0]
        try:
            counters[key] = float(value) if "." in value else int(value)
        except ValueError:
            counters[key] = value

    return counters


def run_benchmark(
    project_dir,
    pinned_core,
    n_val,
    batch_size,
    channels,
):
    cmd = [
        "taskset",
        "-c",
        str(pinned_core),
        "./build/bench_kernel",
        str(n_val),
        str(batch_size),
        str(channels),
    ]
    subprocess.run(cmd, cwd=str(project_dir), check=True)
    return _load_metrics_temp(project_dir)


def run_perf(
    project_dir,
    pinned_core,
    n_val,
    batch_size,
    channels,
    perf_events,
):
    perf_cmd = [
        "taskset",
        "-c",
        str(pinned_core),
        "perf",
        "stat",
        "-x,",
        "-e",
        perf_events,
        "./build/bench_kernel",
        str(n_val),
        str(batch_size),
        str(channels),
    ]
    result = subprocess.run(
        perf_cmd,
        cwd=str(project_dir),
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    stderr_text = result.stderr or ""
    if result.returncode != 0:
        raise RuntimeError(
            f"perf stat failed for N={n_val} with events '{perf_events}': {stderr_text.strip()}"
        )

    perf_run_data = _load_metrics_temp(project_dir)
    perf_counters = _parse_perf_counters(stderr_text)
    return perf_run_data, perf_counters


def log_progress(log_path, message):
    ts = datetime.now().strftime("%H:%M:%S")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{ts}] {message}\n")


def main():
    if len(sys.argv) != 4:
        print("Usage: worker.py <job_id> <user> <description>", file=sys.stderr)
        return 1

    job_id, user, desc = sys.argv[1], sys.argv[2], sys.argv[3]
    job_dir = Path(f"/opt/aos/workspaces/{job_id}")
    artifact_dir = Path("/opt/aos/artifacts")
    history_file = artifact_dir / "history.jsonl"
    workspace_tar = job_dir / "workspace.tar.gz"

    run_log = job_dir / "run.log"
    build_log = job_dir / "build.log"

    try:
        build_log.write_text("", encoding="utf-8")
        run_log.write_text("", encoding="utf-8")

        log_progress(
            run_log, f"Worker started for job_id={job_id}, user={user}, desc={desc}"
        )
        log_progress(run_log, "Extracting workspace archive...")
        with tarfile.open(workspace_tar, "r:gz") as tf:
            tf.extractall(path=job_dir)

        # config.py is expected in the extracted workspace root.
        sys.path.insert(0, str(job_dir))
        import config as app_config

        project_dir = job_dir / "turbogator"
        if not project_dir.is_dir():
            raise FileNotFoundError(
                "turbogator directory missing from submitted workspace"
            )

        log_progress(run_log, "Workspace extracted. Starting build...")

        run_logged(
            ["make", "clean"],
            build_log,
            cwd=project_dir,
            progress_log=run_log,
            label="make clean",
        )
        run_logged(
            ["make", f"-j{app_config.BUILD_JOBS}", "all"],
            build_log,
            cwd=project_dir,
            progress_log=run_log,
            label="make all",
        )
        log_progress(run_log, "Build completed.")

        llvm_output = project_dir / "llvm_mca.txt"
        log_progress(run_log, "Running llvm-mca static analysis...")
        with run_log.open("a", encoding="utf-8") as log_file:
            log_file.write("Running static analysis (llvm-mca)...\n")
            mca_cmd = [
                "llvm-mca",
                *_llvm_mca_flags(app_config),
                str(project_dir / "build/bench_kernel.s"),
            ]
            log_file.write(f"$ {' '.join(mca_cmd)}\n")
            mca_start = datetime.now()
            try:
                mca = subprocess.run(
                    mca_cmd,
                    cwd=str(project_dir),
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    text=True,
                    check=False,
                    timeout=LLVM_MCA_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"llvm-mca timed out after {LLVM_MCA_TIMEOUT_S}s")
            mca_elapsed = (datetime.now() - mca_start).total_seconds()
            log_file.write(
                f"llvm-mca exit={mca.returncode} elapsed={mca_elapsed:.1f}s\n"
            )
            if mca.returncode != 0:
                raise RuntimeError("llvm-mca failed")
            llvm_output.write_text(mca.stdout, encoding="utf-8")
        log_progress(run_log, "llvm-mca completed.")

        representative = str(app_config.REPRESENTATIVE_N)
        log_progress(
            run_log, f"Running callgrind at representative N={representative}..."
        )
        run_logged(
            [
                "taskset",
                "-c",
                str(app_config.PINNED_CPU_CORE),
                "valgrind",
                "--tool=callgrind",
                f"--callgrind-out-file=callgrind.out.{representative}",
                "./build/bench_kernel_dbg",
                representative,
                str(app_config.BATCH_SIZE),
                str(app_config.CHANNELS),
            ],
            run_log,
            cwd=project_dir,
            progress_log=run_log,
            label=f"callgrind representative N={representative}",
        )
        log_progress(run_log, "Callgrind completed.")

        metrics = []
        perf_events = _perf_event_list(app_config)
        run_benchmark_enabled = bool(getattr(app_config, "RUN_BENCHMARK", True))
        run_perf_enabled = bool(getattr(app_config, "RUN_PERF", False))

        if not run_benchmark_enabled and not run_perf_enabled:
            raise RuntimeError("Both RUN_BENCHMARK and RUN_PERF are disabled")

        mode_label = "benchmark+perf"
        if run_benchmark_enabled and not run_perf_enabled:
            mode_label = "benchmark"
        if run_perf_enabled and not run_benchmark_enabled:
            mode_label = "perf"
        log_progress(
            run_log,
            f"Starting benchmark sweep over {len(app_config.SIZES)} sizes in mode: {mode_label}",
        )
        if run_perf_enabled:
            log_progress(run_log, f"Perf events: {perf_events}")

        for N in app_config.SIZES:
            log_progress(
                run_log,
                f"Profiling N={N} (batch={app_config.BATCH_SIZE}, channels={app_config.CHANNELS})",
            )

            run_data = None

            if run_benchmark_enabled:
                run_data = run_benchmark(
                    project_dir,
                    app_config.PINNED_CPU_CORE,
                    N,
                    app_config.BATCH_SIZE,
                    app_config.CHANNELS,
                )
                log_progress(run_log, f"Benchmark execution finished for N={N}.")
                log_progress(run_log, f"Collected cycles metrics for N={N}.")

            if run_perf_enabled:
                perf_run_data, perf_counters = run_perf(
                    project_dir,
                    app_config.PINNED_CPU_CORE,
                    N,
                    app_config.BATCH_SIZE,
                    app_config.CHANNELS,
                    perf_events,
                )

                if run_data is None:
                    run_data = perf_run_data
                else:
                    run_data["cycles_perfrun"] = perf_run_data.get("cycles")

                run_data["perf_counters"] = perf_counters
                log_progress(run_log, f"Collected perf counters for N={N}.")

            if run_data is None:
                raise RuntimeError(f"No run data produced for N={N}")

            metrics.append(run_data)

        metrics_json = {
            "user": user,
            "description": desc,
            "job_id": job_id,
            "data": metrics,
        }

        (job_dir / "metrics.json").write_text(
            json.dumps(metrics_json), encoding="utf-8"
        )

        with history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_json) + "\n")
        log_progress(run_log, "Saved metrics and appended to server history.")
        log_progress(run_log, "Preparing artifact payload.")
        log_progress(run_log, "Job completed successfully.")

        payload_dir = job_dir / "artifact_payload"
        payload_dir.mkdir(parents=True, exist_ok=True)

        for file_name in ["metrics.json", "run.log", "build.log"]:
            src = job_dir / file_name
            shutil.copy2(src, payload_dir / file_name)

        if llvm_output.exists():
            shutil.copy2(llvm_output, payload_dir / "llvm_mca.txt")

        for callgrind_file in project_dir.glob("callgrind.out.*"):
            shutil.copy2(callgrind_file, payload_dir / callgrind_file.name)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_tar = artifact_dir / f"{job_id}.tar.gz"
        with tarfile.open(artifact_tar, "w:gz") as tf:
            for item in payload_dir.iterdir():
                tf.add(item, arcname=item.name)

        return 0
    except Exception as exc:
        with run_log.open("a", encoding="utf-8") as log_file:
            log_file.write(f"ERROR: {exc}\n")
        return 1
    finally:
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
