#!/usr/bin/env python3
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_pipeline.recording.rerun_visualizer import RerunEpisodeReader, RerunLogger


def main():
    parser = argparse.ArgumentParser(description="Rebuild a playable rerun.rrd from an existing episode directory.")
    parser.add_argument("episode_path", help="Path like /path/to/episode_0000 or /path/to/episode_0000/rerun.rrd")
    parser.add_argument("--no-viewer", action="store_true", help="Do not spawn live Rerun viewer while rebuilding")
    args = parser.parse_args()

    episode_path = os.path.abspath(args.episode_path)
    if os.path.isfile(episode_path):
        episode_dir = os.path.dirname(episode_path)
    else:
        episode_dir = episode_path

    if not os.path.isdir(episode_dir):
        raise FileNotFoundError(f"episode_dir not found: {episode_dir}")

    episode_name = os.path.basename(episode_dir.rstrip("/"))
    if not episode_name.startswith("episode_"):
        raise ValueError(f"episode_dir should end with episode_xxxx, got: {episode_name}")
    episode_idx = int(episode_name.split("_")[-1])
    task_dir = os.path.dirname(episode_dir)
    rrd_path = os.path.join(episode_dir, "rerun_rebuilt.rrd")

    reader = RerunEpisodeReader(task_dir=task_dir)
    episode_data = reader.return_episode_data(episode_idx)

    logger = RerunLogger(
        prefix="online/",
        IdxRangeBoundary=60,
        memory_limit="300MB",
        rrd_path=rrd_path,
        spawn_viewer=not args.no_viewer,
    )
    try:
        logger.log_episode_data(episode_data)
    finally:
        logger.close()

    print(rrd_path)


if __name__ == "__main__":
    main()
