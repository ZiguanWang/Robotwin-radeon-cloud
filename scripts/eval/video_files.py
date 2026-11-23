"""Find and name RoboTwin evaluation videos from their episode records."""

from pathlib import Path
from typing import Mapping


def video_filename(row: Mapping) -> str:
    outcome = "success" if bool(row["success"]) else "failure"
    return f"episode{int(row['episode_index'])}_seed{int(row['seed'])}_{outcome}.mp4"


def legacy_filename(row: Mapping) -> str:
    return f"episode{int(row['episode_index'])}.mp4"


def find_video(directory: Path, row: Mapping) -> Path:
    """Locate either the seed-labelled name or an older episode-only name."""
    for name in (video_filename(row), legacy_filename(row)):
        matches = list(directory.rglob(name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"More than one video named {name} under {directory}")
    raise FileNotFoundError(f"Video for seed {row['seed']} under {directory}")


def name_videos(directory: Path, rows: list[dict]) -> list[Path]:
    """Rename completed videos after episode success is known."""
    named = []
    for row in rows:
        source = find_video(directory, row)
        target = source.with_name(video_filename(row))
        if source != target:
            if target.exists():
                raise RuntimeError(f"Target video already exists: {target}")
            source.rename(target)
        named.append(target)
    if len(named) != len(set(named)) or len(list(directory.rglob("*.mp4"))) != len(rows):
        raise RuntimeError(f"Unexpected video count under {directory}")
    return named
