import json
import pathlib
import sys
import tempfile
import time
import unittest
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class _DummyLogger:
    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop


def _install_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _import_episode_writer():
    stubbed_modules = {
        "cv2": dict(
            COLOR_BGRA2BGR=0,
            imwrite=lambda *_args, **_kwargs: True,
            cvtColor=lambda frame, *_args, **_kwargs: frame,
        ),
        "numpy": dict(
            int16=int,
            save=lambda *_args, **_kwargs: None,
        ),
        "zmq": {},
        "logging_mp": dict(getLogger=lambda *_args, **_kwargs: _DummyLogger()),
        "teleop.utils.rerun_visualizer": dict(RerunLogger=type("RerunLogger", (), {})),
    }
    original_modules = {name: sys.modules.get(name) for name in stubbed_modules}
    try:
        for name, attrs in stubbed_modules.items():
            _install_stub(name, **attrs)
        from teleop.utils.episode_writer import EpisodeWriter

        return EpisodeWriter
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class EpisodeWriterControlExtrasCompatTest(unittest.TestCase):
    def test_add_item_accepts_control_extras_without_serializing_it(self):
        EpisodeWriter = _import_episode_writer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = EpisodeWriter(task_dir=tmp_dir, rerun_log=False)
            try:
                self.assertTrue(writer.create_episode())
                writer.add_item(
                    colors={},
                    depths={},
                    states={"left_arm": {"qpos": [], "qvel": [], "torque": []}},
                    actions={"left_arm": {"qpos": [], "qvel": [], "torque": []}},
                    timestamps={"sample_monotonic_ns": 123456789},
                    control_extras={"arm_tauff": [float(i) for i in range(14)]},
                )
                writer.item_data_queue.join()
                writer.save_episode()
                writer.item_data_queue.join()
                while not writer.is_ready():
                    time.sleep(0.01)
            finally:
                writer.close()

            data_path = pathlib.Path(writer.episode_dir) / "data.json"
            payload = json.loads(data_path.read_text())
            self.assertEqual(len(payload["data"]), 1)
            self.assertNotIn("control_extras", payload["data"][0])

    def test_cancel_episode_reuses_episode_index(self):
        EpisodeWriter = _import_episode_writer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = EpisodeWriter(task_dir=tmp_dir, rerun_log=False)
            try:
                self.assertTrue(writer.create_episode())
                first_episode_dir = pathlib.Path(writer.episode_dir)
                first_episode_name = first_episode_dir.name

                writer.cancel_episode()
                self.assertTrue(writer.is_ready())
                self.assertFalse(first_episode_dir.exists())

                self.assertTrue(writer.create_episode())
                self.assertEqual(pathlib.Path(writer.episode_dir).name, first_episode_name)
            finally:
                writer.cancel_episode()
                writer.close()


if __name__ == "__main__":
    unittest.main()
