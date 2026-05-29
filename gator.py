#!/usr/bin/env python3
import argparse
import base64
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

os.environ.setdefault("FORCE_COLOR", "1")
from termcolor import colored

import config as app_config

AUTH = app_config.AUTH
URL = app_config.URL
_BUILD_DONE = False


PINNED_RESULTS = {
    "ezgatr": "reference",
    "baseline": "baseline",
}


_RAW_RED = "\033[91m"
_RAW_RESET = "\033[0m"


class Log:
    @classmethod
    def _ts(cls):
        return datetime.now().strftime("%H:%M:%S")

    @classmethod
    def _stamp(cls):
        return colored(f"[{cls._ts()}]", "blue", force_color=True)

    @classmethod
    def info(cls, msg):
        print(f"{cls._stamp()} {colored(msg, 'blue', force_color=True)}")

    @classmethod
    def success(cls, msg):
        print(f"{cls._stamp()} {colored(msg, 'green', force_color=True)}")

    @classmethod
    def error(cls, msg):
        print(f"{cls._stamp()} {colored(msg, 'red', force_color=True)}")


def run_cmd(cmd, env=None):
    Log.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        Log.error(f"Process exited with code {e.returncode}")
        sys.exit(e.returncode)


# pytorch/numpy thread isolation
BENCH_SINGLE_THREAD_ENV = {
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _unreliable_banner():
    bar = "=" * 80
    print(bar)
    print("DO NOT RELY ON THESE ANYMORE - THEY ARE NOT FACTUAL".center(80))
    print("Use `gator submit <desc>` with config.py SIZES=[1]".center(80))
    print(bar)


class _red_section:
    def __enter__(self):
        sys.stdout.write(_RAW_RED)
        sys.stdout.flush()
        self._saved = {n: Log.__dict__[n] for n in ("info", "success", "error")}

        def _plain(cls, msg):
            print(f"[{cls._ts()}] {msg}")

        for n in ("info", "success", "error"):
            setattr(Log, n, classmethod(_plain))
        return self

    def __exit__(self, *a):
        for n, v in self._saved.items():
            setattr(Log, n, v)
        sys.stdout.write(_RAW_RESET)
        sys.stdout.flush()


def _api_request(path, data=None, headers=None):
    req = urllib.request.Request(f"{URL}{path}", data=data)
    req.add_header("User-Agent", "curl/7.81.0")
    b64 = base64.b64encode(AUTH.encode()).decode()
    req.add_header("Authorization", f"Basic {b64}")

    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    try:
        with urllib.request.urlopen(req) as res:
            return res.read()
    except urllib.error.HTTPError as e:
        Log.error(f"HTTP {e.code}: {e.reason} ({path})")
        sys.exit(1)


def _api_submit(user, desc, tar_path):
    boundary = "----BoundaryTurboGator"
    with open(tar_path, "rb") as f:
        tar_data = f.read()

    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="user"\r\n\r\n{user}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="description"\r\n\r\n{desc}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="workspace"; filename="workspace.tar.gz"\r\nContent-Type: application/gzip\r\n\r\n'.encode(),
            tar_data,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }

    return json.loads(_api_request("/submit", data=body, headers=headers))


def _append_history(root, data):
    path = root / "results/history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def do_build():
    global _BUILD_DONE
    if _BUILD_DONE:
        return
    cpu_count = os.cpu_count() or 1
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count)
    shutil.rmtree("build/skbuild", ignore_errors=True)
    for p in Path("turbogator").glob("*.so"):
        p.unlink()
    run_cmd(["uv", "sync", "--group", "local", "--reinstall-package", "turbogator"])
    _BUILD_DONE = True


def cmd_debug(args):
    with _red_section():
        do_build()
        cmd_validate(args, build=False)
        cmd_microbench(args, build=False)


def cmd_validate(args, build=True):
    with _red_section():
        _unreliable_banner()
        if build:
            do_build()
        Log.info("Validating Turbogator C++ extensions against PyTorch...")
        run_cmd(["uv", "run", "python", "turbogator/validate.py"])
        Log.success("Validation Complete!")
        _unreliable_banner()


def cmd_microbench(args, build=True):
    with _red_section():
        _unreliable_banner()
        if build:
            do_build()
        out_dir = Path("results/microbench")
        out_dir.mkdir(parents=True, exist_ok=True)

        desc = getattr(args, "description", "turbogator")
        t_dim, c_in = app_config.get_dimensions(app_config.REPRESENTATIVE_N)

        flamegraph_path = out_dir / "profile.speedscope.json"
        torch_profile_out = out_dir / "pytorch_profile.trace.json"
        metrics_out = out_dir / f"{desc}_metrics.json"

        bench_env = {**os.environ, **BENCH_SINGLE_THREAD_ENV}

        Log.info(f"Recording py-spy ({desc}): T={t_dim}, C_in={c_in}")
        cmd = [
            "py-spy",
            "record",
            "-o",
            str(flamegraph_path),
            "--format",
            "speedscope",
            "--rate",
            "100",
            "--native",
            "--",
            sys.executable,
            "turbogator/benchmark.py",
            "--desc",
            desc,
            "--t",
            str(t_dim),
            "--c",
            str(c_in),
            "--profile",
            "none",
        ]
        Log.info(f"Running: {' '.join(cmd)}")
        # py-spy exits with code 1 on Linux due to a ECHILD race after the profiled
        # process exits normally; treat it as success if the output file was written.
        result = subprocess.run(cmd, env=bench_env)
        if result.returncode != 0 and not flamegraph_path.exists():
            Log.error(f"py-spy failed with code {result.returncode}")
            sys.exit(result.returncode)

        Log.info("Running torch profiler...")
        run_cmd(
            [
                sys.executable,
                "turbogator/benchmark.py",
                "--desc",
                desc,
                "--t",
                str(t_dim),
                "--c",
                str(c_in),
                "--profile",
                "torch",
                "--profile-out",
                str(torch_profile_out),
            ],
            env=bench_env,
        )

        Log.info("Running clean benchmark pass for metrics...")
        run_cmd(
            [
                sys.executable,
                "turbogator/benchmark.py",
                "--desc",
                desc,
                "--t",
                str(t_dim),
                "--c",
                str(c_in),
                "--warmup",
                "5",
                "--steps",
                "5",
                "--profile",
                "none",
                "--out",
                str(metrics_out),
            ],
            env=bench_env,
        )

        Log.success(f"Microbenchmark complete. Outputs saved to {out_dir}")
        Log.info(f"Metrics saved to {metrics_out.name}")
        Log.info("Open py-spy flamegraph at https://www.speedscope.app/")
        Log.info("Open torch trace at https://ui.perfetto.dev")
        _unreliable_banner()


def cmd_fetch(args):
    job_id = args.job_id
    root = Path(".")

    Log.info(f"Checking server for Job ID: {job_id}")

    status_res = json.loads(_api_request(f"/status/{job_id}"))
    status = status_res.get("status", "unknown")

    if status != "completed":
        Log.error(f"Job is currently: {status}")
        Log.info("Fetching latest logs:")
        logs = _api_request(f"/logs/{job_id}").decode().splitlines()
        for line in logs:
            if line.strip():
                print(f"{Log._stamp()} {line}")
        Log.info("Run fetch again later to check if it has completed.")
        sys.exit(0)

    out_dir = root / "results" / f"raw/{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    Log.info("Job completed! Downloading artifacts...")
    tmp_archive = root / f"tmp_{job_id}.tar.gz"

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            tar_data = _api_request(f"/download/{job_id}")
            with open(tmp_archive, "wb") as f:
                f.write(tar_data)

            with tarfile.open(tmp_archive, "r:gz") as tf:
                tf.extractall(path=out_dir)
            break
        except Exception as exc:
            tmp_archive.unlink(missing_ok=True)
            if attempt == max_attempts:
                Log.error(f"Failed to extract artifacts: {exc}")
                sys.exit(1)
            Log.info("Artifacts incomplete; retrying download...")
            time.sleep(2 * attempt)

    tmp_archive.unlink(missing_ok=True)

    metrics_file = out_dir / "metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        if "error" in data:
            Log.error("Server reported execution failure. Check run.log.")
        else:
            _append_history(root, data)
            Log.success("Metrics appended to local history.")

    Log.success(f"Recovery Complete! Data saved to {out_dir}")


def cmd_clean(args):
    rm_dirs = [
        "build",
        "csrc/build",
        ".skbuild",
        ".cmake",
        "dist",
    ]
    rm_files = [
        "CMakeCache.txt",
        "CMakeInit.txt",
        "cmake_install.cmake",
        "build.ninja",
        "rules.ninja",
        "workspace.tar.gz",
    ]

    for d in rm_dirs:
        shutil.rmtree(d, ignore_errors=True)
    for f in rm_files:
        Path(f).unlink(missing_ok=True)

    for p in Path(".").glob("workspace.*.tar.gz"):
        p.unlink(missing_ok=True)
    for p in Path(".").glob("tmp_*.tar.gz"):
        p.unlink(missing_ok=True)

    for p in Path(".").rglob("__pycache__"):
        if ".venv" not in p.parts and ".git" not in p.parts:
            shutil.rmtree(p, ignore_errors=True)

    for ext in ["*.pyc", "*.egg-info", "*.so", "*.o"]:
        for p in Path(".").rglob(ext):
            if ".venv" not in p.parts and ".git" not in p.parts:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)

    Log.success("Workspace cleaned.")


def cmd_plot(args):
    if not Path("results/history.jsonl").exists():
        Log.error("results/history.jsonl not found.")
        sys.exit(1)
    run_cmd(["uv", "run", "--project", ".", "--group", "local", "analysis/plot_all.py"])
    Log.success("Plots generated in results/plots/")


def cmd_submit(args):
    desc = args.description
    pinned_dir = PINNED_RESULTS.get(desc)
    root = Path(".")

    # server does this now, DO NOT do it locally anymore!!!
    # # sanity check
    # Log.info("Building and validating locally before submit...")
    # do_build()
    # cmd_validate(args, build=False)

    def filter_tar(info):
        name = info.name
        if (
            "build" in name
            or "__pycache__" in name
            or name.endswith(".so")
            or name.endswith(".o")
            or name.startswith(".git")
        ):
            return None
        return info

    fd, tar_name = tempfile.mkstemp(
        prefix="workspace.", suffix=".tar.gz", dir=str(root)
    )
    os.close(fd)
    tar_path = Path(tar_name)

    try:
        with tarfile.open(tar_path, "w:gz") as tf:
            for p in [
                "turbogator",
                "csrc",
                "config.py",
                "pyproject.toml",
                "uv.lock",
                "CMakeLists.txt",
                "README.md",
            ]:
                if Path(p).exists():
                    tf.add(p, filter=filter_tar)

        user = os.environ.get("USER", "unknown")
        Log.info(f"Submitting workspace for '{desc}'...")
        res = _api_submit(user, desc, tar_path)
    finally:
        tar_path.unlink(missing_ok=True)

    job_id = res.get("job_id")

    if not job_id:
        Log.error("Server did not return a Job ID.")
        sys.exit(1)

    Log.success(f"Queued Job ID: {job_id}")
    last_line = 0
    last_status = ""

    while True:
        status_res = json.loads(_api_request(f"/status/{job_id}"))
        status = status_res.get("status", "unknown")

        if status != last_status:
            Log.info(f"Status: {status}")
            last_status = status

        logs = _api_request(f"/logs/{job_id}").decode().splitlines()
        if len(logs) > last_line:
            for line in logs[last_line:]:
                if line.strip():
                    print(f"{Log._stamp()} {line}")
                else:
                    print()
            last_line = len(logs)

        if status in ["completed", "failed", "error", "cancelled"]:
            break
        time.sleep(2)

    out_dir = root / "results" / (pinned_dir if pinned_dir else f"raw/{job_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    Log.info("Downloading artifacts...")
    tmp_archive = root / f"tmp_{job_id}.tar.gz"
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        tar_data = _api_request(f"/download/{job_id}")
        with open(tmp_archive, "wb") as f:
            f.write(tar_data)

        try:
            with tarfile.open(tmp_archive, "r:gz") as tf:
                tf.extractall(path=out_dir)
            break
        except (EOFError, tarfile.ReadError) as exc:
            tmp_archive.unlink(missing_ok=True)
            if attempt == max_attempts:
                Log.error(f"Failed to extract artifacts: {exc}")
                sys.exit(1)
            Log.info("Artifacts incomplete; retrying download...")
            time.sleep(2 * attempt)

    tmp_archive.unlink(missing_ok=True)

    metrics_file = out_dir / "metrics.json"
    advisor_dir = out_dir / "advisor_results"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        if "error" in data:
            Log.error("Server reported execution failure. Check run.log.")
            sys.exit(1)
        else:
            _append_history(root, data)
            Log.success("Metrics appended to local history.")

    if advisor_dir.exists():
        Log.success(f"Intel Advisor results saved to {advisor_dir}")

    Log.success(f"Job Complete! Data saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="TurboGator Dev Workflow")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_debug = subs.add_parser("debug", help="Build and run local validation")
    p_debug.add_argument(
        "description",
        nargs="?",
        default="turbogator",
        help="Model description for microbench (e.g., ezgatr)",
    )
    subs.add_parser("validate", help="Run local validation")
    p_microbench = subs.add_parser("microbench", help="Run local microbenchmarks")
    p_microbench.add_argument(
        "description",
        nargs="?",
        default="turbogator",
        help="Model description for benchmark.py (e.g., ezgatr)",
    )
    subs.add_parser("clean", help="Remove all build and temp files")
    subs.add_parser("plot", help="Generate analysis plots")

    p_sub = subs.add_parser("submit", help="Submit benchmark job to server")
    p_sub.add_argument(
        "description", help="Job description (use 'ezgatr' for reference)"
    )

    p_fetch = subs.add_parser("fetch", help="Retrieve results for a disconnected job")
    p_fetch.add_argument("job_id", help="The Job ID to retrieve (e.g., 17156942_lmg)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.cmd == "debug":
        cmd_debug(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "microbench":
        cmd_microbench(args)
    elif args.cmd == "clean":
        cmd_clean(args)
    elif args.cmd == "plot":
        cmd_plot(args)
    elif args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "fetch":
        cmd_fetch(args)


if __name__ == "__main__":
    main()
