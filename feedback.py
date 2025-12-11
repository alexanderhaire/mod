"""
Feedback logging utilities.
Captures per-answer ratings to a JSONL file for later analysis or fine-tuning.
"""

import base64
import hashlib
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


FEEDBACK_FILE = "feedback_log.jsonl"
FEEDBACK_ATTACHMENTS_DIR = Path("feedback_attachments")
MAX_ATTACHMENT_COUNT = 5
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MB per attachment
MAX_INLINE_ATTACHMENT_BYTES = 300_000  # Limit inline base64 to keep logs compact


class _MemoryAttachment:
    """Minimal file-like wrapper so pasted bytes can flow through the normal saver."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def save_feedback_attachments(files: Iterable[Any] | None) -> Tuple[list[str], list[str]]:
    """
    Persist uploaded feedback attachments to disk.
    Returns (saved_paths, errors) so callers can surface any skips to the user.
    """
    saved: list[str] = []
    errors: list[str] = []

    if not files:
        return saved, errors

    FEEDBACK_ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, file in enumerate(files):
        if idx >= MAX_ATTACHMENT_COUNT:
            errors.append("Maximum attachment count reached; extra files were skipped.")
            break

        try:
            data = file.getvalue()
        except Exception as err:  # noqa: BLE001
            errors.append(f"Could not read an attachment: {err}")
            continue

        if not data:
            errors.append("Skipped an empty attachment.")
            continue

        if len(data) > MAX_ATTACHMENT_BYTES:
            errors.append("Skipped an attachment larger than 5 MB.")
            continue

        suffix = Path(file.name).suffix.lower() or ".bin"
        filename = f"fb_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
        destination = FEEDBACK_ATTACHMENTS_DIR / filename

        try:
            destination.write_bytes(data)
            saved.append(str(destination))
        except Exception as err:  # noqa: BLE001
            errors.append(f"Could not save attachment {filename}: {err}")

    return saved, errors


def attachments_from_paste(payload: Dict[str, Any] | None) -> Tuple[list[Any], list[str]]:
    """
    Decode pasted images (data URLs) into in-memory attachments for saving.
    Returns (file_like_objects, errors).
    """
    files: list[Any] = []
    errors: list[str] = []

    if not payload or not isinstance(payload, dict):
        return files, errors

    images = payload.get("images")
    if not isinstance(images, list):
        return files, errors

    for idx, image in enumerate(images):
        data_url = image.get("data_url") if isinstance(image, dict) else None
        if not data_url or not isinstance(data_url, str) or not data_url.startswith("data:image"):
            errors.append("Ignored a pasted item that was not an image.")
            continue

        try:
            _, b64_data = data_url.split(",", 1)
        except ValueError:
            errors.append("Pasted image data was malformed.")
            continue

        try:
            data = base64.b64decode(b64_data)
        except Exception as err:  # noqa: BLE001
            errors.append(f"Could not decode a pasted image: {err}")
            continue

        if not data:
            errors.append("Skipped an empty pasted image.")
            continue

        if len(data) > MAX_ATTACHMENT_BYTES:
            errors.append("Skipped a pasted image larger than 5 MB.")
            continue

        name_hint = image.get("name") if isinstance(image, dict) else None
        filename = name_hint if name_hint else f"pasted_image_{idx}.png"
        files.append(_MemoryAttachment(filename, data))

    return files, errors


def _attachment_payload(path: str) -> Dict[str, Any]:
    """
    Build a serializable payload that a multimodal model can consume.
    Includes small inline base64 data so the model can reason about the image without file I/O.
    """
    record: Dict[str, Any] = {"path": path}
    p = Path(path)
    record["name"] = p.name
    record["mime"] = mimetypes.guess_type(p.name)[0] if p.name else None

    if not p.exists():
        record["error"] = "Attachment file missing on disk."
        return record

    try:
        data = p.read_bytes()
    except Exception as err:  # noqa: BLE001
        record["error"] = f"Could not read attachment: {err}"
        return record

    record["size_bytes"] = len(data)
    record["sha256"] = hashlib.sha256(data).hexdigest()

    if len(data) <= MAX_INLINE_ATTACHMENT_BYTES:
        record["inline_base64"] = base64.b64encode(data).decode("ascii")
        record["encoding"] = "base64"
    else:
        record["inline_base64"] = None
        record["encoding"] = None
        record["truncated"] = True

    return record


def log_feedback(event: Dict[str, Any]) -> None:
    """
    Append a feedback event to disk.
    Expected keys include: user, chat_id, question, answer, helpful (bool), notes, sql, attachments.
    """
    payload = dict(event)
    payload.setdefault("timestamp", int(time.time()))

    attachments = payload.get("attachments")
    if attachments:
        payload["attachments_meta"] = [_attachment_payload(path) for path in attachments]

    line = json.dumps(payload, ensure_ascii=True)
    # Ensure directory exists if a path is provided.
    folder = os.path.dirname(FEEDBACK_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")
