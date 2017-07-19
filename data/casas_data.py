import cPickle
import gzip
import numpy as np

def test_format():
    # Test the format of mnist
    dataset = '/home/tra161/WORK/Data/rtdrbm/data/mnist/data.pkl.gz'
    data_dict = cPickle.load(gzip.open(dataset, 'rb'))
#    print "Dataset: %s\n" % (data_dict['name'])
    for k,_ in data_dict.iteritems():
        print(k)

    print(data_dict['label_dict'])
    print(data_dict['name'])
    print(data_dict['n_classes'])
    
          
    for k,v in data_dict['X'].iteritems():
        print(k)

    print(len(data_dict['X']['train']))

def make_casas_data(clength=1):
    data_dict={'X':{},'y':{}}
    data_dict['X']['train'] = []
    data_dict['y']['train'] = []
    data_dict['X']['verify'] = []
    data_dict['y']['verify'] = []
    data_dict['X']['test'] = []
    data_dict['y']['test'] = []
    dat_dir = "/home/tra161/WORK/Data/CASAS/adlmr/non_temporal/clength"+str(clength)
    nclass = 0
    for fold in range(1,27):
        dat_train = np.genfromtxt(dat_dir+"/"+"fold"+str(fold)+"_train_data.csv")
        lab_train = np.genfromtxt(dat_dir+"/"+"fold"+str(fold)+"_train_label.csv").astype(int)
        dat_valid = np.genfromtxt(dat_dir+"/"+"fold"+str(fold)+"_valid_data.csv")
        lab_valid = np.genfromtxt(dat_dir+"/"+"fold"+str(fold)+"_valid_label.csv").astype(int)

        
        cl = max(np.amax(lab_train),np.max(lab_valid))
        if cl>nclass:
            nclass = cl

        data_dict['X']['train'].append(dat_train)
        data_dict['y']['train'].append(lab_train)
        data_dict['X']['verify'].append(dat_valid)
        data_dict['y']['verify'].append(lab_valid)
        data_dict['X']['test'].append(dat_valid)
        data_dict['y']['test'].append(lab_valid)

    data_dict['name']='casas'
    data_dict['n_classes'] = nclass+1
    print(type(data_dict['n_classes']))
    save_file = dat_dir + "/" + "data.pkl.gz"
    cPickle.dump(data_dict,gzip.open(save_file,"wb"))
    
if __name__=="__main__":
    make_casas_data(clength=3)
    #test_format()
