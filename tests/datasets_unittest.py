import cPickle
import gzip
import os
import unittest

DATA_HOME="data"

class TestDatasets(unittest.TestCase):
    """Class to test the presence and integrity of various datasets
    """
    def setUp(self):
        pass

    supervised_datasets = ['mnist', 'ocr', 'ocr-mini', 'usps']
    unsupervised_datasets = ['nottingham']

    # NOTE: Some other test that have not yet been implemented
    #   1. Check if the dataset contains equal number of and at least one fold.
    #   2. Check if n_classes equals the number of classes in data.
    #   3. Check for validity of format of the data matrices.
    
    # NOTE: Re-think the reasons for carrying out the tests. Do we need all the
    # datasets to be present? Can we continue even if the test for one dataset
    # fails?

    def test_supervised_datasets(self):
        """Check if all supervised datasets are in order.

        Input
        -----

        Output
        ------

        """
        for dataset in self.supervised_datasets:
            print "checking dataset %s..." % (dataset)
            # Check if dataset exists
            try:
                data = cPickle.load(gzip.open(os.path.join(DATA_HOME, dataset, 
                                              'data.pkl.gz'), 'rb'))
            except IOError:
                print '\tdataset folder %s/%s does not exist' \
                        % (DATA_HOME, dataset)

            print '\tdataset %s exists' % (dataset)

            # Check if dataset dictionary has a name key, and if not add it
            if not data.has_key('name'):
                data['name'] = dataset
                cPickle.dump(data, gzip.open(os.path.join(DATA_HOME, dataset, 
                                                          'data.pkl.gz'), 
                                             'rb'))
                print '\tadded name key to %d dataset' % (dataset)

            # Check if the dataset dictionary contains mandatory keys
            mandatory_keys = ['X', 'y', 'n_classes']
            for k in mandatory_keys:
                assert data.has_key(k), \
                        '\tsupervised dataset %s does not have the key %s' \
                        % (dataset, k)
                print '\tdataset %s contains mandatory key %s' % (dataset, k)

            # Check if the dataset dictionary contains optional keys
            optional_keys = ['label_dict']
            for k in optional_keys:
                if not data.has_key(k):
                    print '\tdataset %s does not contain optional key %s' \
                            % (dataset, k)
            else:
                print '\tdataset %s contains optional key %s' % (dataset, k)

            print ''


    def test_unsupervised_datasets(self):
        """Check if all supervised datasets are in order.

        Input
        -----

        Output
        ------

        """
        for dataset in self.unsupervised_datasets:
            print "checking dataset %s..." % (dataset)
            # Check if dataset exists
            try:
                data = cPickle.load(gzip.open(os.path.join(DATA_HOME, dataset,
                                              'data.pkl.gz'), 'rb'))
            except IOError:
                print '\tdataset folder %s/%s does not exist' \
                        % (DATA_HOME, dataset)

            print '\tdataset %s exists' % (dataset)

            # Check if dataset dictionary has a name key, and if not add it
            if not data.has_key('name'):
                data['name'] = dataset
                cPickle.dump(data, gzip.open(os.path.join(DATA_HOME, dataset,
                                                          'data.pkl.gz'), 
                                             'rb'))
                print '\tadded name key to %d dataset' % (dataset)

            # Check if the dataset dictionary contains mandatory keys
            mandatory_keys = ['X', 'n_dims']
            for k in mandatory_keys:
                assert data.has_key(k), \
                        '\tsupervised dataset %s does not have the key %s' \
                        % (dataset, k)
                print '\tdataset %s contains mandatory key %s' % (dataset, k)

            # Check if the dataset dictionary contains optional keys
            optional_keys = ['label_dict']
            for k in optional_keys:
                if not data.has_key(k):
                    print '\tdataset %s does not contain optional key %s' \
                            % (dataset, k)
            else:
                print '\tdataset %s contains optional key %s' % (dataset, k)

            print ''


if __name__ == '__main__':
    unittest.main()
