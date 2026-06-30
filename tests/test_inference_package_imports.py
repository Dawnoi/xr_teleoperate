import unittest


class InferencePackageImportsTest(unittest.TestCase):
    def test_new_inference_package_exports_existing_api(self):
        from teleop.inference.online_session import OnlineInferenceSession
        from teleop.inference.pi05_protocol import build_pi05_observation_payload
        from teleop.inference.pose_transform import PoseTransformer
        from teleop.inference.protocol import TcpJsonTransport

        self.assertIsNotNone(OnlineInferenceSession)
        self.assertIsNotNone(build_pi05_observation_payload)
        self.assertIsNotNone(PoseTransformer)
        self.assertIsNotNone(TcpJsonTransport)


if __name__ == "__main__":
    unittest.main()
