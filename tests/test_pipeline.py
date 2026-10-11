"""Comprobaciones de invariantes: dimensiones, ausencia de fuga e inferencia."""
import unittest

import numpy as np
import pandas as pd

from src.dataset import curate_and_split
from src.features import extract_image_features, feature_names
from src.preprocessing import ImageConfig, normalize_roi, preprocess, resize_max
from src.experiments import fit_transforms, transform_blocks


class PipelineTests(unittest.TestCase):
    def test_resize_preserves_small_images_and_ratio(self):
        small = np.zeros((80, 120, 3), dtype=np.uint8)
        self.assertEqual(resize_max(small).shape, small.shape)
        self.assertEqual(resize_max(np.zeros((800,1200,3),np.uint8)).shape, (427,640,3))

    def test_padding_has_no_mask_pixels(self):
        rgb, mask = normalize_roi(np.ones((20,40,3),np.uint8), np.ones((20,40),np.uint8))
        self.assertEqual(rgb.shape, (128,128,3))
        self.assertEqual(int(mask.sum()), 128*64)
        self.assertEqual(int(mask[:32].sum()), 0)

    def test_uniform_image_fallback_and_feature_dimensions(self):
        rgb = np.full((91,117,3),128,np.uint8)
        f = extract_image_features(rgb)
        for name, names in feature_names().items():
            self.assertEqual(len(f[name]),len(names))
            self.assertTrue(np.isfinite(f[name]).all())
        self.assertEqual(f['shape'][-1], 1)
        self.assertEqual(len(preprocess(rgb)['regions']), 0)

    def test_local_regions_have_valid_masks(self):
        rgb = np.zeros((120,180,3),np.uint8)
        rgb[30:90,60:120] = 255
        result = preprocess(rgb, ImageConfig(filter_name='none'))
        self.assertGreater(len(result['regions']),0)
        for region in result['regions']:
            self.assertTrue(region['mask'].any())
            self.assertEqual(region['rgb'].shape,(128,128,3))

    def test_conflicts_excluded_and_splits_reproducible(self):
        rows = []
        for label in ('a','b'):
            for i in range(20):
                rows.append(dict(image_id=f'{label}{i}', relative_path=f'{label}/{i:02d}.png',
                                 label=label, valid=True, pixel_sha256=f'{label}{i}'))
        rows.extend([dict(rows[0],image_id='copy',relative_path='a/zz.png'),
                     dict(rows[1],image_id='conflict',relative_path='b/conflict.png',label='b')])
        frame = pd.DataFrame(rows)
        result = curate_and_split(frame)
        pd.testing.assert_frame_equal(result,curate_and_split(frame.sample(frac=1,random_state=4)))
        self.assertEqual((result.exclusion_reason=='etiquetas_contradictorias').sum(),2)
        self.assertEqual((result.exclusion_reason=='copia_repetida').sum(),1)
        selected=result[result.included]
        self.assertTrue(selected.pixel_sha256.is_unique)
        self.assertTrue((pd.crosstab(selected.label,selected.split)>0).all().all())

    def test_scaler_pca_fit_only_training(self):
        rng=np.random.default_rng(42)
        blocks={'gabor':rng.normal(size=(15,24)), 'hog':rng.normal(size=(15,32))}
        blocks['gabor'][10:]+=1000
        train=np.arange(15)<10
        transforms, transformed=fit_transforms(blocks,train,hog_components=4)
        np.testing.assert_allclose(transforms['gabor']['scaler'].mean_,blocks['gabor'][:10].mean(axis=0))
        self.assertEqual(transforms['hog']['pca'].n_samples_,10)
        applied=transform_blocks(blocks,transforms)
        for name in blocks:
            np.testing.assert_allclose(applied[name],transformed[name],rtol=1e-5,atol=1e-4)


if __name__=='__main__':
    unittest.main()
