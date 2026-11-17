import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scf_project import read_dataset, specifications, SEED
from sklearn.model_selection import train_test_split

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X,cls.y=read_dataset()
    def test_dataset_shape_and_target(self):
        self.assertEqual(self.X.shape,(140,3))
        self.assertEqual(self.y.shape,(140,))
        self.assertAlmostEqual(self.y.min(),1.6)
        self.assertAlmostEqual(self.y.max(),2.14)
    def test_geometry_levels_and_unique_combinations(self):
        self.assertEqual([len(np.unique(self.X[:,i])) for i in range(3)],[4,5,7])
        self.assertEqual(len(np.unique(self.X,axis=0)),140)
    def test_no_ANN_feature_leakage(self):
        self.assertEqual(self.X.shape[1],3)
        self.assertTrue(np.isfinite(self.X).all())
        self.assertTrue(np.isfinite(self.y).all())
    def test_80_20_fixed_split_disjoint(self):
        indices=np.arange(140)
        tr,te=train_test_split(indices,test_size=.2,random_state=SEED,shuffle=True)
        self.assertEqual((len(tr),len(te)),(112,28))
        self.assertTrue(set(tr).isdisjoint(set(te)))
    def test_estimators_all_fit_on_training_slice(self):
        indices=np.arange(140)
        tr,te=train_test_split(indices,test_size=.2,random_state=SEED,shuffle=True)
        for name,estimator,params in specifications():
            with self.subTest(model=name):
                estimator.fit(self.X[tr],self.y[tr])
                self.assertEqual(estimator.predict(self.X[te[:2]]).shape,(2,))

if __name__=='__main__': unittest.main()
