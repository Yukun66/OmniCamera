import unittest
from pathlib import Path

from infer import load_and_validate


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODES = {
    "t2v",
    "i2v",
    "v2v",
    "traj2v_t2v",
    "traj2v_i2v",
    "traj2v_v2v",
    "r2v_t2v",
    "r2v_i2v",
    "r2v_v2v",
}


class ExampleConfigTests(unittest.TestCase):
    def test_all_configs_are_valid_and_cover_every_mode(self):
        modes = {
            load_and_validate(path)["mode"]
            for path in sorted((ROOT / "examples").glob("*.json"))
        }
        self.assertEqual(modes, EXPECTED_MODES)


if __name__ == "__main__":
    unittest.main()
