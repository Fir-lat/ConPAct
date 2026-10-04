from __future__ import annotations

import argparse
import base64
import filecmp
import os
import tempfile
import json
from pathlib import Path


def convert(record, max_frames=3):
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("A sample requires a request and a complete assistant reply")
    if messages[-1].get("role") != "assistant" or messages[-2].get("role") != "user":
        raise ValueError(
            "The final user message must be followed by one assistant target"
        )
    if record.get("loss", [0] * (len(messages) - 1) + [1]) != [0] * (
        len(messages) - 1
    ) + [1]:
        raise ValueError("Only the final assistant reply may receive loss")
    converted, images = [], []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError("Unsupported message role")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") == "image_url" and role == "user":
                    value = part["image_url"]["url"]
                    header, separator, body = value.partition(",")
                    if not separator or header not in (
                        "data:image/png;base64",
                        "data:image/jpeg;base64",
                        "data:image/jpg;base64",
                        "data:image/webp;base64",
                    ):
                        raise ValueError(
                            "Expected an embedded image from the recorded request"
                        )
                    if not base64.b64decode(body, validate=True):
                        raise ValueError("Empty image")
                    images.append(value)
                    parts.append("<image>")
                else:
                    raise ValueError("Unsupported message content")
            content = "".join(parts)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty or invalid message text")
        converted.append({"role": role, "content": content})
    if images and record.get("images"):
        raise ValueError("Images must use one representation per sample")
    if not images:
        images = record.get("images", [])
        if not isinstance(images, list) or not all(
            isinstance(value, str) for value in images
        ):
            raise ValueError("Invalid images field")
    if not 0 <= len(images) <= max_frames:
        raise ValueError("Sample exceeds the visible-frame limit")
    if sum(message["content"].count("<image>") for message in converted) != len(images):
        raise ValueError("Image placeholders and ordered images do not match")
    return {"messages": converted, "images": images}


def prepare(source, destination, role="joint", max_frames=3):
    if role not in ("joint", "planner", "actor") or max_frames < 1:
        raise ValueError("Invalid role or frame limit")
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Output must differ from the input dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = None
    try:
        with source.open(encoding="utf-8") as input_file:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix="." + destination.name + ".",
                delete=False,
            ) as output_file:
                temporary = Path(output_file.name)
                for number, line in enumerate(input_file, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if role != "joint":
                        meta = record.get("meta", {})
                        recorded_role = meta.get("role", meta.get("type"))
                        if recorded_role not in ("planner", "actor"):
                            raise ValueError(
                                f"Row {number} has no planner/actor role metadata"
                            )
                        if recorded_role != role:
                            continue
                    row = convert(record, max_frames)
                    for index, image in enumerate(row["images"]):
                        if not image.startswith("data:image/"):
                            path = Path(image)
                            path = (
                                path
                                if path.is_absolute()
                                else source.resolve().parent / path
                            )
                            if not path.is_file():
                                raise ValueError(
                                    f"Row {number} references a missing image"
                                )
                            row["images"][index] = str(path.resolve())
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                if not count:
                    raise ValueError("No samples for the selected role")
        if destination.exists():
            if not filecmp.cmp(temporary, destination, shallow=False):
                raise FileExistsError(
                    "Prepared dataset changed; use a new experiment directory"
                )
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--role", choices=("joint", "planner", "actor"), default="joint"
    )
    parser.add_argument("--max-frames", type=int, default=3)
    args = parser.parse_args()
    count = prepare(args.input, args.output, args.role, args.max_frames)
    print(json.dumps({"role": args.role, "rows": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
