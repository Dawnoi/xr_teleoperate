"""Serialized post-recording validation jobs for raw episodes."""

from __future__ import annotations

from queue import Queue
from threading import Lock, Thread
from pathlib import Path
from typing import Any, Callable

from data_pipeline.audit.episode_validation import (
    DEFAULT_ACTION_SEMANTICS_CONFIG,
    DEFAULT_TIME_ALIGNMENT_CONFIG,
    validate_finalized_episode,
    write_validation_report,
)


class EpisodeValidationManager:
    """Runs one validation job at a time outside the teleoperation control loop."""

    def __init__(
        self,
        *,
        validator: Callable[[str | Path], dict[str, Any]] | None = None,
        action_semantics_config: str | Path = DEFAULT_ACTION_SEMANTICS_CONFIG,
        time_alignment_config: str | Path = DEFAULT_TIME_ALIGNMENT_CONFIG,
    ) -> None:
        self._validator = validator or (
            lambda episode_dir: validate_finalized_episode(
                episode_dir,
                action_semantics_config=action_semantics_config,
                time_alignment_config=time_alignment_config,
            )
        )
        self._queue: Queue[Path | None] = Queue()
        self._lock = Lock()
        self._pending_episode_dirs: list[str] = []
        self._current_episode_dir = ""
        self._last_report: dict[str, Any] = {}
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

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._current_episode_dir or self._pending_episode_dirs)

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending = list(self._pending_episode_dirs)
            current = self._current_episode_dir
            last_report = dict(self._last_report)
        return {
            "pending": bool(current or pending),
            "current_episode_dir": current,
            "queued_episode_dirs": pending,
            "last_validation": last_report,
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
            report = self._validator(episode_dir)
            write_validation_report(episode_dir, report)
            with self._lock:
                self._current_episode_dir = ""
                self._last_report = report
            self._queue.task_done()
