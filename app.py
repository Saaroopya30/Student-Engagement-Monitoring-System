import json
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from pipeline import PROJECT_DIR, ProcessingCancelled, process_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

WEB_DIR = PROJECT_DIR / "web_runs"
UPLOAD_DIR = WEB_DIR / "uploads"
WEB_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOB_STATUS = {}
JOB_LOCK = threading.Lock()


def allowed_file(filename):
    return Path(filename).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}


def set_job_status(job_id, **updates):
    with JOB_LOCK:
        current = JOB_STATUS.get(job_id, {}).copy()
        current.update(updates)
        JOB_STATUS[job_id] = current


def process_job(run_id, input_path, output_path, run_dir, filename):
    try:
        set_job_status(
            run_id,
            run_id=run_id,
            state="queued",
            stage_label="Queued",
            percent=0,
            processed_frames=0,
            total_frames=0,
            eta_seconds=None,
            message="Waiting to start analysis.",
            original_filename=filename,
            cancel_requested=False,
            started_at=time.time(),
        )

        def on_progress(update):
            stage_map = {
                "processing": "Processing Video",
                "reporting": "Generating Reports",
                "completed": "Completed",
            }
            set_job_status(
                run_id,
                state=update["stage"],
                stage_label=stage_map.get(update["stage"], update["stage"].title()),
                percent=update["percent"],
                processed_frames=update["processed_frames"],
                total_frames=update["total_frames"],
                eta_seconds=update.get("eta_seconds"),
                message=update["message"],
                original_filename=filename,
            )

        results = process_video(
            input_path=input_path,
            output_path=output_path,
            report_dir=run_dir,
            display=False,
            max_frames=None,
            progress_callback=on_progress,
            should_cancel=lambda: JOB_STATUS.get(run_id, {}).get("cancel_requested", False),
        )

        metadata = {
            "run_id": run_id,
            "original_filename": filename,
            "processed_frames": results["processed_frames"],
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        set_job_status(
            run_id,
            state="completed",
            stage_label="Completed",
            percent=100,
            processed_frames=results["processed_frames"],
            total_frames=JOB_STATUS.get(run_id, {}).get("total_frames", 0),
            eta_seconds=0,
            message="Analysis complete. Redirecting to results.",
            original_filename=filename,
        )
    except ProcessingCancelled as exc:
        for path in [run_dir / "annotated.mp4", run_dir / "engagement_results.db", run_dir / "engagement_results.csv"]:
            if path.exists():
                path.unlink()
        for name in [
            "summary.json",
            "student_summary.csv",
            "engagement_trend.png",
            "engagement_distribution.png",
            "dashboard.html",
            "metadata.json",
        ]:
            candidate = run_dir / name
            if candidate.exists():
                candidate.unlink()
        set_job_status(
            run_id,
            state="cancelled",
            stage_label="Cancelled",
            percent=100,
            eta_seconds=None,
            message=str(exc),
            original_filename=filename,
        )
    except Exception as exc:
        set_job_status(
            run_id,
            state="failed",
            stage_label="Failed",
            percent=100,
            eta_seconds=None,
            message=str(exc),
            original_filename=filename,
        )


@app.route("/", methods=["GET"])
def index():
    runs = sorted(
        [path for path in WEB_DIR.iterdir() if path.is_dir() and path.name != "uploads"],
        reverse=True,
    )[:8]
    recent_runs = []
    for run_dir in runs:
        summary_path = run_dir / "summary.json"
        metadata_path = run_dir / "metadata.json"
        if not summary_path.exists() or not metadata_path.exists():
            continue
        recent_runs.append(
            {
                "run_id": run_dir.name,
                "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
            }
        )
    return render_template("index.html", recent_runs=recent_runs)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("video")
    if file is None or file.filename == "":
        abort(400, "Please upload a video file.")
    if not allowed_file(file.filename):
        abort(400, "Unsupported file type.")

    run_id = uuid.uuid4().hex[:10]
    run_dir = WEB_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(file.filename)
    input_path = UPLOAD_DIR / f"{run_id}_{filename}"
    output_path = run_dir / "annotated.mp4"
    file.save(input_path)
    worker = threading.Thread(
        target=process_job,
        args=(run_id, input_path, output_path, run_dir, filename),
        daemon=True,
    )
    worker.start()
    return redirect(url_for("progress", run_id=run_id))


@app.route("/progress/<run_id>", methods=["GET"])
def progress(run_id):
    if run_id not in JOB_STATUS and not (WEB_DIR / run_id).exists():
        abort(404)
    return render_template("progress.html", run_id=run_id)


@app.route("/status/<run_id>", methods=["GET"])
def status(run_id):
    status_payload = JOB_STATUS.get(run_id)
    if status_payload is None:
        metadata_path = WEB_DIR / run_id / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status_payload = {
                "run_id": run_id,
                "state": "completed",
                "stage_label": "Completed",
                "percent": 100,
                "processed_frames": metadata.get("processed_frames", 0),
                "total_frames": metadata.get("processed_frames", 0),
                "eta_seconds": 0,
                "message": "Analysis complete. Your reports are ready.",
                "original_filename": metadata.get("original_filename", "Uploaded video"),
            }
        else:
            abort(404)
    return jsonify(status_payload)


@app.route("/cancel/<run_id>", methods=["POST"])
def cancel(run_id):
    if run_id not in JOB_STATUS:
        abort(404)

    current = JOB_STATUS.get(run_id, {})
    if current.get("state") in {"completed", "failed", "cancelled"}:
        return jsonify(current)

    set_job_status(
        run_id,
        cancel_requested=True,
        state="cancelling",
        stage_label="Cancelling",
        message="Cancellation requested. Stopping analysis safely.",
        eta_seconds=None,
    )
    return jsonify(JOB_STATUS[run_id])


@app.route("/results/<run_id>", methods=["GET"])
def result(run_id):
    run_dir = WEB_DIR / run_id
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    if not run_dir.exists() or not summary_path.exists() or not metadata_path.exists():
        abort(404)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return render_template("result.html", run_id=run_id, summary=summary, metadata=metadata)


@app.route("/files/<run_id>/<path:filename>", methods=["GET"])
def files(run_id, filename):
    run_dir = WEB_DIR / run_id
    if not run_dir.exists():
        abort(404)
    kwargs = {"as_attachment": False, "conditional": True}
    if filename.lower().endswith(".mp4"):
        kwargs["mimetype"] = "video/mp4"
    return send_from_directory(run_dir, filename, **kwargs)


if __name__ == "__main__":
    app.run(debug=True)
