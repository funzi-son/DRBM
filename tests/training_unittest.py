import cPickle
import gzip
import os
import unittest

from collections import OrderedDict
from config import Config
from train_models import grid_search

DATA_HOME=os.path.abspath('data')
EXAMPLES_HOME=os.path.abspath('examples')

class TestModels(unittest.TestCase):
    """Test routines for training various models."""
    def setUp(self):
        pass
    

    def test_all_models(self):
        """
        """
        # This list specifies which models to evaluate, using which optimisers
        # and on which datasets. In order to evaluate any new model, optimiser
        # and dataset add it to this list.
        dat_mod_opt_list = [('drbm', 'usps', 'gd'), 
                            ('hdrbm', 'usps', 'gd'), 
                            ('rbm', 'usps', 'gd'), 
                            ('rnndrbm', 'ocr-mini', 'gd'), 
                            ('rnnnade', 'nottingham', 'gd'),
                            ('rtdrbm', 'ocr-mini', 'gd'),
                            ('drbm', 'usps', 'adadelta'), 
                            ('hdrbm', 'usps', 'adadelta'), 
                            ('rbm', 'usps', 'adadelta'), 
                            ('rnndrbm', 'ocr-mini', 'adadelta'), 
                            ('rnnnade', 'nottingham', 'adadelta'),
                            ('rtdrbm', 'ocr-mini', 'adadelta'),
                            ('drbm', 'usps', 'rmsprop'), 
                            ('hdrbm', 'usps', 'rmsprop'), 
                            ('rbm', 'usps', 'rmsprop'), 
                            ('rnndrbm', 'ocr-mini', 'rmsprop'), 
                            ('rnnnade', 'nottingham', 'rmsprop'),
                            ('rtdrbm', 'ocr-mini', 'rmsprop')]

        for (m, d, o) in dat_mod_opt_list:
            # Create log-file

            # Load dataset list
            dat_f = os.path.join(EXAMPLES_HOME, 'datasets', d+'-dataset.txt')
            f_dataset = open(dat_f, 'r')
            dataset = [ds.strip() for ds in f_dataset.readlines()][0]
            dat_dict = cPickle.load(gzip.open(dataset, 'rb'))
            out_path = os.path.dirname(dataset)
            f_dataset.close()

            # Load model configuration
            mod_f = os.path.join(EXAMPLES_HOME, 'models', m+'.cfg')
            f_config = file(mod_f)
            mod_dict = OrderedDict(Config(f_config))  # Duck-typing

            # Load optimiser configuration
            opt_f = os.path.join(EXAMPLES_HOME, 'optimisers', o+'.cfg')
            f_config = file(opt_f)
            opt_dict = OrderedDict(Config(f_config))  # Duck-typing

            print ("Training the %s model on the %s dataset using the %s"
                   "optimiser" % (m, d, o))

            grid_search(mod_dict, opt_dict, dat_dict, os.path.join(out_path,
                                                                'models'))


if __name__ == '__main__':
    unittest.main()
