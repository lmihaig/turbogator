#!/usr/bin/env python3
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

AUTH = "turbogator:TurboGator2026"
URL = "https://aos.licu.dev"


class Log:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    @classmethod
    def _ts(cls):
        return datetime.now().strftime("%H:%M:%S")

    @classmethod
    def info(cls, msg):
        print(f"[{cls._ts()}] {cls.BLUE} {msg}{cls.RESET}")

    @classmethod
    def success(cls, msg):
        print(f"[{cls._ts()}] {cls.GREEN} {msg}{cls.RESET}")

    @classmethod
    def error(cls, msg):
        print(f"[{cls._ts()}] {cls.RED} {msg}{cls.RESET}")


def run_cmd(cmd):
    Log.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        Log.error(f"Process exited with code {e.returncode}")
        sys.exit(e.returncode)


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


def cmd_debug(args):
    shutil.rmtree("build/skbuild", ignore_errors=True)
    for p in Path("turbogator").glob("*.so"):
        p.unlink()

    run_cmd(["uv", "sync", "--reinstall-package", "turbogator"])

    Log.info("Validating Turbogator C++ extensions against PyTorch...")
    run_cmd(["uv", "run", "python", "turbogator/validate.py"])
    Log.success("Validation Complete!")


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
    run_cmd(
        ["uv", "run", "--project", ".", "--group", "analysis", "analysis/plot_all.py"]
    )
    Log.success("Plots generated in results/plots/")


def cmd_submit(args):
    desc = args.description
    is_ref = desc == "ezgatr"
    root = Path(".")
    tar_path = root / "workspace.tar.gz"

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
                    print(f"{Log.BLUE}[{Log._ts()}]{Log.RESET} {line}")
                else:
                    print()
            last_line = len(logs)

        if status in ["completed", "failed", "error", "cancelled"]:
            break
        time.sleep(2)

    out_dir = root / "results" / ("reference" if is_ref else f"raw/{job_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    Log.info("Downloading artifacts...")
    tar_data = _api_request(f"/download/{job_id}")
    tmp_archive = root / f"tmp_{job_id}.tar.gz"

    with open(tmp_archive, "wb") as f:
        f.write(tar_data)

    with tarfile.open(tmp_archive, "r:gz") as tf:
        tf.extractall(path=out_dir)

    tmp_archive.unlink()
    tar_path.unlink()

    metrics_file = out_dir / "metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        if "error" in data:
            Log.error("Server reported execution failure. Check run.log.")
            sys.exit(1)
        elif not is_ref:
            with open(root / "results/history.jsonl", "a") as f:
                f.write(json.dumps(data) + "\n")
            Log.success("Metrics appended to local history.")

    Log.success(f"Job Complete! Data saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="TurboGator Dev Workflow")
    subs = parser.add_subparsers(dest="cmd", required=True)

    subs.add_parser("debug", help="Build and run local validation")
    subs.add_parser("clean", help="Remove all build and temp files")
    subs.add_parser("plot", help="Generate analysis plots")

    p_sub = subs.add_parser("submit", help="Submit benchmark job to server")
    p_sub.add_argument(
        "description", help="Job description (use 'ezgatr' for reference)"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.cmd == "debug":
        cmd_debug(args)
    elif args.cmd == "clean":
        cmd_clean(args)
    elif args.cmd == "plot":
        cmd_plot(args)
    elif args.cmd == "submit":
        cmd_submit(args)


if __name__ == "__main__":
    main()
