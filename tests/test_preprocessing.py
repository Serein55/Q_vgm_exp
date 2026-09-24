import unittest

import numpy as np

from qvgm.envs.libero_env import observation_for_policy, quat_to_axisangle


class ObservationTests(unittest.TestCase):
    def test_rotation_and_state_layout_without_mutation(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        q = np.array([0.0, 0.0, 0.0, 1.00000001])
        obs = dict(
            agentview_image=image,
            robot0_eye_in_hand_image=image,
            robot0_eef_pos=np.array([1, 2, 3]),
            robot0_eef_quat=q,
            robot0_gripper_qpos=np.array([0.1, -0.1]),
        )
        result = observation_for_policy(obs, "pick up")
        np.testing.assert_array_equal(result["observation/image"], image[::-1, ::-1])
        np.testing.assert_allclose(result["observation/state"], [1, 2, 3, 0, 0, 0, 0.1, -0.1])
        self.assertEqual(q[3], 1.00000001)
        self.assertTrue(result["observation/image"].flags.c_contiguous)

    def test_axis_angle(self):
        np.testing.assert_allclose(quat_to_axisangle([0, 0, 1, 0]), [0, 0, np.pi])


if __name__ == "__main__":
    unittest.main()
