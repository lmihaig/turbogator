import os
import subprocess
import time

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

app = FastAPI()
WORKSPACE_DIR = "/opt/aos/workspaces"
ARTIFACT_DIR = "/opt/aos/artifacts"


@app.post("/submit")
async def submit_job(
    user=Form(...),
    description=Form(...),
    workspace: UploadFile = File(...),
):
    job_id = f"{int(time.time())}_{user}"
    job_path = f"{WORKSPACE_DIR}/{job_id}"
    os.makedirs(job_path, exist_ok=True)

    tar_path = f"{job_path}/workspace.tar.gz"
    with open(tar_path, "wb") as buffer:
        buffer.write(await workspace.read())

    subprocess.run(
        ["tsp", "python3", "/opt/aos/scripts/worker.py", job_id, user, description],
        check=False,
    )

    return {"status": "queued", "job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id):
    if os.path.exists(f"{ARTIFACT_DIR}/{job_id}.tar.gz"):
        return {"status": "completed"}
    return {"status": "processing_or_queued"}


@app.get("/logs/{job_id}", response_class=PlainTextResponse)
async def get_logs(job_id):
    log_path = f"/opt/aos/workspaces/{job_id}/run.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return f.read()
    return "Initializing job...\n"


def remove_artifact(file_path):
    time.sleep(5)
    if os.path.exists(file_path):
        os.remove(file_path)


@app.get("/download/{job_id}")
async def download_artifact(job_id, background_tasks: BackgroundTasks):
    file_path = f"{ARTIFACT_DIR}/{job_id}.tar.gz"

    # background_tasks.add_task(remove_artifact, file_path)

    return FileResponse(path=file_path, filename=f"results_{job_id}.tar.gz")
