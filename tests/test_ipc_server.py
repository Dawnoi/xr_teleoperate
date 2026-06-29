import pathlib
import sys
import types
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "zmq" not in sys.modules:
    sys.modules["zmq"] = types.SimpleNamespace(
        Context=types.SimpleNamespace(instance=lambda: None),
        Poller=lambda: None,
        POLLIN=1,
        NOBLOCK=1,
        error=types.SimpleNamespace(ContextTerminated=RuntimeError),
    )

from teleop.operator.ipc import IPC_Server


class IPCServerMessageHandlingTest(unittest.TestCase):
    def _make_server(self, received_keys):
        server = object.__new__(IPC_Server)
        server.on_press = received_keys.append
        server.get_state = lambda: {}
        return server

    def test_cmd_go_home_maps_to_h(self):
        received_keys = []
        server = self._make_server(received_keys)

        reply = IPC_Server._handle_message(
            server,
            {
                "reqid": "req-go-home",
                "cmd": "CMD_GO_HOME",
            },
        )

        self.assertEqual(reply, {"repid": "req-go-home", "status": "ok", "msg": "ok"})
        self.assertEqual(received_keys, ["h"])

    def test_unsupported_command_is_rejected_explicitly(self):
        received_keys = []
        server = self._make_server(received_keys)

        reply = IPC_Server._handle_message(
            server,
            {
                "reqid": "req-bad",
                "cmd": "CMD_UNKNOWN",
            },
        )

        self.assertEqual(reply["repid"], "req-bad")
        self.assertEqual(reply["status"], "error")
        self.assertIn("cmd not supported", reply["msg"])
        self.assertEqual(received_keys, [])


if __name__ == "__main__":
    unittest.main()
