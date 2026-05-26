import os
import secrets
import subprocess
import time

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config import AUTH

ROOT_DIR = os.environ.get("AOS_ROOT_DIR", "/opt/aos")
WORKSPACE_DIR = os.environ.get("AOS_WORKSPACE_DIR", f"{ROOT_DIR}/workspaces")
ARTIFACT_DIR = os.environ.get("AOS_ARTIFACT_DIR", f"{ROOT_DIR}/artifacts")
SCRIPTS_DIR = os.environ.get("AOS_SCRIPTS_DIR", f"{ROOT_DIR}/scripts")

_EXPECTED_USER, _, _EXPECTED_PASS = AUTH.partition(":")

os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

app = FastAPI()
_security = HTTPBasic()


def require_auth(creds: HTTPBasicCredentials = Depends(_security)):
    ok_user = secrets.compare_digest(creds.username, _EXPECTED_USER)
    ok_pass = secrets.compare_digest(creds.password, _EXPECTED_PASS)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.post("/submit")
async def submit_job(
    user=Form(...),
    description=Form(...),
    workspace: UploadFile = File(...),
    _=Depends(require_auth),
):
    job_id = f"{int(time.time())}_{user}"
    job_path = f"{WORKSPACE_DIR}/{job_id}"
    os.makedirs(job_path, exist_ok=True)

    tar_path = f"{job_path}/workspace.tar.gz"
    with open(tar_path, "wb") as buffer:
        buffer.write(await workspace.read())

    subprocess.run(
        ["tsp", "python3", f"{SCRIPTS_DIR}/worker.py", job_id, user, description],
        check=False,
    )

    return {"status": "queued", "job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id, _=Depends(require_auth)):
    if os.path.exists(f"{ARTIFACT_DIR}/{job_id}.tar.gz"):
        return {"status": "completed"}
    return {"status": "processing_or_queued"}


@app.get("/logs/{job_id}", response_class=PlainTextResponse)
async def get_logs(job_id, _=Depends(require_auth)):
    log_path = f"{WORKSPACE_DIR}/{job_id}/run.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return f.read()
    return "Initializing job...\n"


def remove_artifact(file_path):
    time.sleep(5)
    if os.path.exists(file_path):
        os.remove(file_path)


@app.get("/download/{job_id}")
async def download_artifact(job_id, background_tasks: BackgroundTasks, _=Depends(require_auth)):
    file_path = f"{ARTIFACT_DIR}/{job_id}.tar.gz"

    # background_tasks.add_task(remove_artifact, file_path)

    return FileResponse(path=file_path, filename=f"results_{job_id}.tar.gz")
