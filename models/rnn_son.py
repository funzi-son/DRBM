"""
Implementation of the RNN 
"""
# Author: Son N. Tran
# sontn.fz@gmail.com

import theano
import theano.tensor as T

class RNN(object):
    def __init__(self,n_input,n_class,hypers,init_params=None):
        """Constructs and compiles Theano functions for training and
        prediction.

        Input
        -----
        n_input : integer
          Number of inputs which make up a part of the visible layer.
        n_class : integer
          Number of classes.
        hypers : dictionary
          Model hyperparameters.
        init_params : list
          Model parameters.
        """

        self.model_type = str(hypers['model_type'])
        self.n_input    = int(n_input)
        self.n_hidden   = int(hypers['n_hidden'])
        self.n_class    = int(n_class)
        self.activation = str(hypers['activation'])
        self.L1         = float(hypers['weight_decay'])
        self.L2_sqr     = float(hypers['weight_decay'])
        self.seed       = float(hypers['seed'])
        

        # build the model graph
        cost,pred,updates_train = build_rnn(n_input,n_hidden,n_class,truncate_len)
        # compute gradient
        lr = T.scalar('learning_rate',dtype=theano.config.floatX)
        grads = T.grad(cost,params)
        updates_train.update(((param,param-lr*grad) for param,grad in zip(params,grads)))
        # functions for training, predicting with and saving the model
        self.loss = theano.function([x,y],cost)
        self.train_step = theano.function([x,y,lr],cost, updates=updates_train,allow_input_downcast=True)
        self.pred_function  = theano.function([x],pred,allow_input_downcast=True)
        self.get_model_parameters = theano.function([],paramsS)

    def total_loss(self,X,Y):
        loss = np.sum([self.loss(x,y) for x,y in zip(X,Y)])
        return loss/np.sum([len(y) for y in Y])
        
def build_rnn(n_input,n_hidden,n_class,truncate_len):
    # Define and initial model's params
    if init_params is None:   
        Whh_init = np.asarray(N_RNG.normal(size=(n_hidden,n_hidden),scale=0.01),dtype=theano.config.floatX)
        Who_init = np.asarray(N_RNG.normal(size=(n_hidden,n_class),scale=0.01),dtype=theano.config.floatX)
        Wvh_init = np.asarray(N_RNG.normal(size=(n_input,n_hidden),scale=0.01),dtype=theano.config.floatX)

        hb_init  = np.zeros((n_hidden,),dtype=theano.config.floatX)
        ob_init  = np.zeros((n_class,),dtype=theano.config.floatX)
    else:
        Whh_init = init_params[0]
        Who_init = init_params[1]
        Wvh_init = init_params[2]

        hb_init  = init_params[3]
        ob_init  = init_params[4]

    Whh = theano.shared(Whh_init,'name'='Whh')
    Who = theano.shared(Who_init,'name'='Who')
    Wvh = theano.shared(Wvh_init,'name'='Wvh')

    hb  = theano.shared(hb_init,'name'='hb')
    ob  = theano.shared(ob_init,'name'='ob')

    params = [Whh,Who,Wvh,hb,ob]

    # inference
    [o,s],updates = theano.scan(
        rnn_cell,
        sequence=x,
        output_info=[None,dict(initial=T.zeros(n_hidden))],
        non_sequences=params,
        truncate_gradient=truncate_len
        )
    pred = T.argmax(o,axis=1)
    cost = T.sum(T.nnet.categorical_crossentropy(o,y))
    
    return cost,pred,updates

def rnn_cell(x_t,h_t_prev,params):
        h_t = T.tanh(params[0].dot(h_t_prev) + params[2].dot(x_t) + hb)
        o_t = T.nnet.softmax(params[1].dot(s_t) + ob)
        return [o_t[0],s_t]
