import cPickle
import gzip

result_file = '/home/tra161/WORK/Data/rtdrbm/data/mnist/models/model-drbm-500-0.0001-ll-0.01-sigmoid-1--1-860331-gd-0.1-linear-9-4-1-0.0-0.0-5-10-1-False-er-1001--eval-offline-na-na-na-na-na-na.pkl.gz'
#result_file = '/home/tra161/WORK/Data/rtdrbm/data/mnist/models/model-drbm-500-0.0001-ll-0.01-sigmoid-1--1-860331-gd-0.1-linear-9-4-1-0.0-0.0-5-10-1-False-er-1001.pkl.gz'
def main():
    rs = cPickle.load(gzip.open(result_file, 'rb'))
    for k,v in rs.iteritems():
        print(k)
    print(rs['test_offline'])
    print(rs['validation'])
          
if __name__=="__main__":
    main()
