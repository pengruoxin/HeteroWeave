import hashlib
import pickle
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "assets" / "component_pool" / "assignment_hybrid_4.pkl"


class ReleaseAssetTest(unittest.TestCase):
    def test_component_pool_checksum_and_positions(self):
        digest = hashlib.sha256(POOL.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "93a8bc3c5d82704216003e42d74341c6739707ce81babdc93a7395de6521ffd0",
        )
        with POOL.open("rb") as handle:
            assignment = pickle.load(handle)
        self.assertEqual(len(assignment.center2block), 4)
        self.assertTrue(all(len(group) > 1 for group in assignment.center2block))

    def test_all_imagenet_paper_models_are_released(self):
        expected = [
            'configs/imagenet/heteroweave_p_10m3g_100e.py',
            'configs/imagenet/heteroweave_e_10m3g_100e.py',
            'configs/imagenet/heteroweave_main_100e.py',
            'configs/imagenet/dery_10m3g_100e.py',
            'configs/imagenet/dery_baseline_100e.py',
            'configs/baselines/imagenet/model_stitching_10m3g_100e.py',
            'configs/baselines/imagenet/model_stitching_30m6g_100e.py',
            'configs/baselines/imagenet/side_tuning_10m3g_100e.py',
            'configs/baselines/imagenet/side_tuning_30m6g_100e.py',
            'configs/baselines/imagenet/snnet_3ti9s_100e.py',
            'configs/baselines/imagenet/snnet_9ti3s_100e.py',
            'configs/references/imagenet/regnet_y_800mf_100e.py',
            'configs/references/imagenet/mobilenet_v3_large_100e.py',
            'configs/references/imagenet/regnet_y_3_2gf_100e.py',
            'configs/references/imagenet/resnet50_100e.py',
            'configs/references/imagenet/swin_tiny_100e.py',
        ]
        self.assertTrue(all((ROOT / path).is_file() for path in expected))

    def test_auxiliary_main_rows_are_released(self):
        expected = {
            'retrieval': {'clip', 'heteroweave_r1', 'heteroweave_r2',
                          'heteroweave_r3'},
            'segmentation': {'resnet50', 'heteroweave_s1', 'heteroweave_s2',
                             'heteroweave_s3', 'heteroweave_s4'},
            'detection': {'resnet50', 'heteroweave_d1', 'heteroweave_d2',
                          'heteroweave_d3'},
        }
        for task, identifiers in expected.items():
            namespace = runpy.run_path(
                ROOT / 'configs' / task / 'paper_models.py')
            observed = {item['id'] for item in namespace['candidates']}
            self.assertTrue(identifiers <= observed)


if __name__ == "__main__":
    unittest.main()
