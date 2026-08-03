"""Serialized post-recording validation jobs for raw episodes."""

from __future__ import annotations

import json
import os
from multiprocessing import get_context
from queue import Queue
from threading import Lock, Thread
from pathlib import Path
from typing import Any, Callable

from data_pipeline.audit.episode_validation import (
    DEFAULT_ACTION_SEMANTICS_CONFIG,
    DEFAULT_MOBILE_TRAINING_CONFIG,
    DEFAULT_TIME_ALIGNMENT_CONFIG,
    validate_finalized_episode,
    write_validation_report,
)


VALIDATION_PROCESS_NICE = 10


def _validate_finalized_episode_in_subprocess(
    episode_dir: str,
    action_semantics_config: str,
    time_alignment_config: str,
    mobile_training_config: str,
) -> None:
    """Run CPU-bound post-record validation outside the teleop interpreter."""
    os.nice(VALIDATION_PROCESS_NICE)
    report = validate_finalized_episode(
        episode_dir,
        action_semantics_config=action_semantics_config,
        time_alignment_config=time_alignment_config,
        mobile_training_config=mobile_training_config,
    )
    write_validation_report(episode_dir, report)


class EpisodeValidationManager:
    """Runs one validation job at a time outside the teleoperation control loop."""

    def __init__(
        self,
        *,
        validator: Callable[[str | Path], dict[str, Any]] | None = None,
        action_semantics_config: str | Path = DEFAULT_ACTION_SEMANTICS_CONFIG,
        time_alignment_config: str | Path = DEFAULT_TIME_ALIGNMENT_CONFIG,
        mobile_training_config: str | Path = DEFAULT_MOBILE_TRAINING_CONFIG,
    ) -> None:
        self._custom_validator = validator
        self._action_semantics_config = str(action_semantics_config)
        self._time_alignment_config = str(time_alignment_config)
        self._mobile_training_config = str(mobile_training_config)
        self._process_context = get_context("spawn")
        self._queue: Queue[Path | None] = Queue()
        self._lock = Lock()
        self._pending_episode_dirs: list[str] = []
        self._current_episode_dir = ""
        self._last_report: dict[str, Any] = {}
        self._validation_process_pid: int | None = None
        self._closed = False
        self._worker = Thread(target=self._run, name="episode-validation", daemon=True)
        self._worker.start()

    def submit(self, episode_dir: str | Path) -> None:
        path = Path(episode_dir)
        with self._lock:
            if self._closed:
                raise RuntimeError("episode validation manager is closed")
            self._pending_episode_dirs.append(str(path))
        self._queue.put(path)

    def validate_after_finalize(self, episode_dir: str | Path) -> dict[str, Any]:
        """Synchronously validate from the episode writer's finalization thread.

        The actual default validator remains a low-priority spawned child, but
        the writer owns its lifecycle: it cannot become ready for a new episode
        until this finalized episode has a durable validation report.
        """
        path = Path(episode_dir)
        with self._lock:
            if self._closed:
                raise RuntimeError("episode validation manager is closed")
            if self._current_episode_dir or self._pending_episode_dirs:
                raise RuntimeError(
                    "episode validation cannot start while another validation job is active: "
                    f"current={self._current_episode_dir!r} pending={self._pending_episode_dirs!r}"
                )
            self._current_episode_dir = str(path)

        if self._custom_validator is not None:
            report = self._custom_validator(path)
            write_validation_report(path, report)
        else:
            report = self._validate_in_subprocess(path)

        with self._lock:
            self._current_episode_dir = ""
            self._last_report = report
        return report

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._current_episode_dir or self._pending_episode_dirs)

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending = list(self._pending_episode_dirs)
            current = self._current_episode_dir
            last_report = dict(self._last_report)
            validation_process_pid = self._validation_process_pid
        return {
            "pending": bool(current or pending),
            "current_episode_dir": current,
            "queued_episode_dirs": pending,
            "last_validation": last_report,
            "validation_process_pid": validation_process_pid,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._worker.join()

    def _run(self) -> None:
        while True:
            episode_dir = self._queue.get()
            if episode_dir is None:
                self._queue.task_done()
                return
            with self._lock:
                self._current_episode_dir = str(episode_dir)
                self._pending_episode_dirs.remove(str(episode_dir))
            if self._custom_validator is not None:
                # Dependency-injected validators are test-only and may be local
                # callables that cannot be serialized by multiprocessing spawn.
                report = self._custom_validator(episode_dir)
                write_validation_report(episode_dir, report)
            else:
                report = self._validate_in_subprocess(episode_dir)
            with self._lock:
                self._current_episode_dir = ""
                self._last_report = report
            self._queue.task_done()

    def _validate_in_subprocess(self, episode_dir: Path) -> dict[str, Any]:
        process = self._process_context.Process(
            target=_validate_finalized_episode_in_subprocess,
            args=(
                str(episode_dir),
                self._action_semantics_config,
                self._time_alignment_config,
                self._mobile_training_config,
            ),
            name="episode-validation",
            daemon=False,
        )
        process.start()
        with self._lock:
            self._validation_process_pid = process.pid
        process.join()
        with self._lock:
            self._validation_process_pid = None
        if process.exitcode != 0:
            raise RuntimeError(
                f"episode validation subprocess failed: episode={episode_dir} exitcode={process.exitcode}"
            )

        report_path = episode_dir / "validation.json"
        if not report_path.is_file():
            raise RuntimeError(f"episode validation subprocess did not create {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise RuntimeError(f"episode validation report must be an object: {report_path}")
        return report
