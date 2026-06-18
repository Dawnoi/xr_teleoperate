import base64
import pathlib
import socket
import sys
import unittest

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class InferenceProtocolTest(unittest.TestCase):
    def test_newline_codec_handles_split_and_coalesced_messages(self):
        from teleop.utils.inference_protocol import NewlineJsonCodec, encode_json_line, make_reset_message

        codec = NewlineJsonCodec()
        message_a = {"type": "action", "seq": 1, "action": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5]]}
        message_b = make_reset_message(reason="unit-test")
        wire_a = encode_json_line(message_a)
        wire_b = encode_json_line(message_b)

        self.assertEqual(codec.feed(wire_a[:9]), [])
        self.assertEqual(codec.feed(wire_a[9:] + wire_b[:5]), [message_a])
        self.assertEqual(codec.feed(wire_b[5:]), [message_b])
        self.assertEqual(wire_b, b'{"type":"reset","reason":"unit-test"}\n')

    def test_parse_left_accepts_action_l_or_single_action(self):
        from teleop.utils.inference_protocol import parse_action_chunk

        step = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 2.0, 0.8]
        for payload in ({"type": "action", "action_l": [step]}, {"type": "action", "action": [step]}):
            with self.subTest(payload=payload):
                parsed = parse_action_chunk(payload, arm_side="left")
                self.assertIsNotNone(parsed.left)
                self.assertIsNone(parsed.right)
                self.assertEqual(len(parsed.left), 1)
                self.assertTrue(np.allclose(parsed.left[0][:3], [0.1, -0.2, 0.3]))
                self.assertTrue(np.allclose(parsed.left[0][3:7], [0.0, 0.0, 0.0, 1.0]))
                self.assertAlmostEqual(parsed.left[0][7], 0.8)

    def test_parse_both_rejects_single_action(self):
        from teleop.utils.inference_protocol import parse_action_chunk

        with self.assertRaises(ValueError):
            parse_action_chunk(
                {"type": "action", "action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]},
                arm_side="both",
            )

    def test_parse_rejects_ambiguous_action_keys(self):
        from teleop.utils.inference_protocol import parse_action_chunk

        with self.assertRaises(ValueError):
            parse_action_chunk(
                {
                    "type": "action",
                    "action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                    "action_l": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                },
                arm_side="left",
            )

    def test_parse_rejects_empty_or_wrong_dim_or_nan(self):
        from teleop.utils.inference_protocol import parse_action_chunk

        bad_payloads = (
            {"type": "action", "action": []},
            {"type": "action", "action": [[0.0, 0.0, 0.0]]},
            {"type": "action", "action": [[0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 1.0, 0.0]]},
            {"type": "action", "action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_action_chunk(payload, arm_side="left")

    def test_parse_normalizes_quaternion(self):
        from teleop.utils.inference_protocol import parse_action_chunk

        parsed = parse_action_chunk(
            {"type": "action", "action_r": [[1.0, 2.0, 3.0, 1.0, 2.0, 2.0, 4.0, -0.5]]},
            arm_side="right",
        )

        self.assertIsNone(parsed.left)
        self.assertIsNotNone(parsed.right)
        quat = parsed.right[0][3:7]
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0)
        self.assertTrue(np.allclose(quat, np.array([1.0, 2.0, 2.0, 4.0]) / 5.0))

    def test_encode_jpeg_base64_roundtrip_decodable(self):
        from teleop.utils.inference_protocol import encode_jpeg_base64

        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :16] = [255, 32, 16]
        image[:, 16:] = [0, 200, 240]

        encoded = encode_jpeg_base64(image)
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, image.shape)
        self.assertEqual(decoded.dtype, np.uint8)
        mean_abs_error = np.abs(decoded.astype(np.int16) - image.astype(np.int16)).mean()
        self.assertLess(mean_abs_error, 8.0)

    def test_make_reset_message(self):
        from teleop.utils.inference_protocol import make_reset_message

        self.assertEqual(make_reset_message(), {"type": "reset"})
        self.assertEqual(make_reset_message(reason="done"), {"type": "reset", "reason": "done"})

    def test_json_codec_rejects_nonstandard_nan_and_encode_disallows_nan(self):
        from teleop.utils.inference_protocol import NewlineJsonCodec, encode_json_line

        with self.assertRaises(ValueError):
            encode_json_line({"type": "observation", "value": float("nan")})

        codec = NewlineJsonCodec()
        with self.assertRaises(ValueError):
            codec.feed(b'{"type":"action","value":NaN}\n')

    def test_tcp_json_transport_uses_newline_json_without_blocking_recv(self):
        from teleop.utils.inference_protocol import TcpJsonTransport

        client_sock, server_sock = socket.socketpair()
        self.addCleanup(server_sock.close)
        transport = TcpJsonTransport.from_socket(client_sock)
        self.addCleanup(transport.close)

        self.assertTrue(transport.is_connected())
        self.assertIsNone(transport.recv_json_nonblocking())

        transport.send_json({"type": "observation", "seq": 1})
        self.assertEqual(server_sock.recv(4096), b'{"type":"observation","seq":1}\n')

        server_sock.sendall(b'{"type":"action","action":[[0,0,0,0,0,0,1,0.05]]}\n')
        self.assertEqual(
            transport.recv_json_nonblocking(),
            {"type": "action", "action": [[0, 0, 0, 0, 0, 0, 1, 0.05]]},
        )
        self.assertIsNone(transport.recv_json_nonblocking())

    def test_tcp_json_transport_reset_sends_reset_message(self):
        from teleop.utils.inference_protocol import TcpJsonTransport

        client_sock, server_sock = socket.socketpair()
        self.addCleanup(server_sock.close)
        transport = TcpJsonTransport.from_socket(client_sock)
        self.addCleanup(transport.close)

        transport.reset(reason="unit-test")

        self.assertEqual(server_sock.recv(4096), b'{"type":"reset","reason":"unit-test"}\n')

    def test_tcp_json_transport_can_discard_stale_pending_rx(self):
        from teleop.utils.inference_protocol import TcpJsonTransport

        client_sock, server_sock = socket.socketpair()
        self.addCleanup(server_sock.close)
        transport = TcpJsonTransport.from_socket(client_sock)
        self.addCleanup(transport.close)

        server_sock.sendall(
            b'{"type":"action","action":[[0,0,0,0,0,0,1,0.01]]}\n'
            b'{"type":"action","action":[[1,0,0,0,0,0,1,0.02]]}\n'
        )
        self.assertEqual(
            transport.recv_json_nonblocking(),
            {"type": "action", "action": [[0, 0, 0, 0, 0, 0, 1, 0.01]]},
        )

        transport.clear_pending_rx()

        self.assertIsNone(transport.recv_json_nonblocking())


if __name__ == "__main__":
    unittest.main()
