import importlib
import sys
import types
import unittest
from unittest import mock


class _DummyLogger:
    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop


def _module_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


RECORDING_MODULES = (
    "teleop.recording.episode_writer",
    "teleop.recording.rerun_visualizer",
    "teleop.utils.lerobot_v2_writer",
)


class RecordingOptionalDependenciesTest(unittest.TestCase):
    def setUp(self):
        pinocchio_stub = _module_stub(
            "pinocchio",
            SE3=type("SE3", (), {}),
            rpy=_module_stub("rpy", matrixToRpy=lambda rotation: [0.0, 0.0, 0.0]),
        )
        parquet_stub = _module_stub("parquet")
        pyarrow_stub = _module_stub(
            "pyarrow",
            field=lambda *args, **kwargs: ("field", args, kwargs),
            schema=lambda fields: ("schema", fields),
            struct=lambda fields: ("struct", fields),
            string=lambda: "string",
            float64=lambda: "float64",
            int64=lambda: "int64",
            list_=lambda dtype: ("list", dtype),
            Codec=type("Codec", (), {"is_available": staticmethod(lambda _codec: False)}),
        )
        self.base_stubs = {
            "cv2": _module_stub(
                "cv2",
                COLOR_BGRA2BGR=0,
                COLOR_BGR2RGB=1,
                COLOR_BGRA2RGBA=2,
                imread=lambda *_args, **_kwargs: None,
                imwrite=lambda *_args, **_kwargs: True,
                cvtColor=lambda frame, *_args, **_kwargs: frame,
            ),
            "logging_mp": _module_stub("logging_mp", getLogger=lambda *_args, **_kwargs: _DummyLogger()),
            "pinocchio": pinocchio_stub,
            "pyarrow": pyarrow_stub,
            "pyarrow.parquet": parquet_stub,
        }
        self._clear_recording_modules()

    def tearDown(self):
        self._clear_recording_modules()

    @staticmethod
    def _clear_recording_modules():
        for module_name in RECORDING_MODULES:
            sys.modules.pop(module_name, None)
        recording_pkg = sys.modules.get("teleop.recording")
        if recording_pkg is not None:
            for attr_name in ("episode_writer", "rerun_visualizer", "lerobot_v2_writer"):
                recording_pkg.__dict__.pop(attr_name, None)
        utils_pkg = sys.modules.get("teleop.utils")
        if utils_pkg is not None:
            utils_pkg.__dict__.pop("lerobot_v2_writer", None)

    def test_recording_modules_import_without_optional_dependencies(self):
        with mock.patch.dict(sys.modules, {**self.base_stubs, "zmq": None, "rerun": None}, clear=False):
            episode_writer = importlib.import_module("teleop.recording.episode_writer")
            rerun_visualizer = importlib.import_module("teleop.recording.rerun_visualizer")

        self.assertTrue(hasattr(episode_writer, "EpisodeWriter"))
        self.assertTrue(hasattr(rerun_visualizer, "RerunLogger"))

    def test_recording_modules_import_with_dependency_stubs_without_specs(self):
        with mock.patch.dict(
            sys.modules,
            {
                **self.base_stubs,
                "zmq": types.ModuleType("zmq"),
                "rerun": types.ModuleType("rerun"),
            },
            clear=False,
        ):
            episode_writer = importlib.import_module("teleop.recording.episode_writer")
            rerun_visualizer = importlib.import_module("teleop.recording.rerun_visualizer")

        self.assertTrue(hasattr(episode_writer, "EpisodeWriter"))
        self.assertTrue(hasattr(rerun_visualizer, "RerunLogger"))
