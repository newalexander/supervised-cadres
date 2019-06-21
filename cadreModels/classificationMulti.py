## kClassification.py
## K-label classification cadres

from __future__ import division, print_function, absolute_import

import time
import numpy as np
import tensorflow as tf

def eNet(alpha, lam, v):
    """Elastic-net regularization penalty"""
    return lam * (alpha * tf.reduce_sum(tf.abs(v)) + 
                  (1-alpha) * tf.reduce_sum(tf.square(v)))
    
class kClassCadreModel(object):
    
    def __init__(self, M=2, gamma=10., lambda_d=0.01, lambda_W=0.01,
                 alpha_d=0.9, alpha_W=0.9, Tmax=10000, record=100, 
                 eta=2e-3, Nba=50, eps=1e-3):
        ## hyperparameters / structure
        self.M = M                # number of cadres
        self.gamma = gamma        # cadre assignment sharpness
        self.lambda_d = lambda_d  # regularization strengths
        self.lambda_W = lambda_W
        self.alpha_d = alpha_d    # elastic net mixing weights
        self.alpha_W = alpha_W    
        self.fitted = False
        ## optimization settings
        self.Tmax = Tmax     # maximum iterations
        self.record = record # record points
        self.eta = eta       # initial stepsize
        self.Nba = Nba       # minibatch size
        self.eps = eps       # convergence tolerance 
        ## parameters
        self.W = 0     # regression weights
        self.W0 = 0    # regression biases
        self.C = 0     # cadre centers
        self.d = 0     # cadre assignment weights
        ## data
        self.X = None       # copy of input data
        self.Y = None       # copy of target values
        self.columns = None # column-names
        ## outputs
        self.loss = [] # loss trajectory
        self.accs = [] # accuracy trajectory
        self.accsVa = [] # validation accuracy trajectory
        self.time = [] # times
    
    def get_params(self, deep=True):
        return {'M': self.M, 'gamma': self.gamma, 'lambda_d': self.lambda_d, 
                'lambda_W': self.lambda_W, 'alpha_d': self.alpha_d, 
                'alpha_W': self.alpha_W, 'Tmax': self.Tmax, 'record': self.record, 
                'eta': self.eta, 'Nba': self.Nba, 'eps': self.eps}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, Xtr, Ytr, Xva=None, Yva=None, names=None, seed=16162, store=False):
        np.random.seed(seed)
        """Fits multilabel classification cadre model"""
        if names is not None:
            self.columns = names
            
        (Ntr, P), K = Xtr.shape, np.unique(Ytr).shape[0]#Ytr.shape[1]
        # number of observations, features, labels
        self.fitted = True
        if store:
            self.X = Xtr
            self.Y = Ytr
        
        ############################################
        ## tensorflow parameters and placeholders ##
        ############################################
        tf.reset_default_graph()
    
        ## cadre centers parameter
        C = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(P,self.M)), 
                            dtype=tf.float32, name='C')
        ## cadre determination weights parameter
        d = tf.Variable(np.random.uniform(size=(P)), dtype=tf.float32, name='d')
        
        ## regression hyperplane weights parameter
        W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(K,P,self.M)), 
                            dtype=tf.float32, name='W')
        ## regression hyperplane bias parameter
        W0 = tf.Variable(tf.zeros(shape=(K,self.M), dtype=tf.float32), 
                             dtype=tf.float32, name='W0')
    
        X = tf.placeholder(dtype=tf.float32, shape=(None,P), name='X')
        Y = tf.placeholder(dtype=tf.int32, shape=(None, ), name='Y')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
              tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, X)), 
              tf.abs(d))
                
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                             tf.expand_dims(t,0))), axis=1), T, name='G')                 

        ## E[n,y,m] = e^m_y(x^n)
        E = tf.add(tf.einsum('np,kpm->nkm', X, W), W0, name='E')
        
        ## F[n,k] = f_k(x^n)
        F = tf.einsum('nm,nkm->nk', G, E, name='F')
        Yhat = tf.argmax(F, axis=1)

        ## L = 1 / N sum_n log(p(y[n] | x[n])) + reg(Theta)
        L = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=F)) + eNet(self.alpha_d, self.lambda_d, d) + eNet(self.alpha_W, self.lambda_W, W)
        opt = tf.train.AdamOptimizer(learning_rate=self.eta)
        optimizer = opt.minimize(L)
        
        ####################
        ## learning model ##
        ####################
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            t0 = time.time()
            ## perform optimization
            for t in range(self.Tmax):
                inds = np.random.choice(Ntr, self.Nba, replace=False)
                sess.run(optimizer, feed_dict={X: Xtr[inds,:], Y: Ytr[inds]})
                # record-keeping        
                if not t % self.record:
#                     if len(self.time):
#                         print(t, self.loss[-1], self.accs[-1], self.accsVa[-1])
#                     else:
#                         print(t)
                    self.time.append(time.time() - t0)
                    l, yhats = sess.run([L, Yhat], feed_dict={X: Xtr, Y: Ytr})
                    self.loss.append(l)
                    self.accs.append(np.mean(yhats == Ytr))
                    if Xva is not None:
                        yhats = Yhat.eval(feed_dict={X: Xva, Y: Yva})
                        self.accsVa.append(np.mean(yhats == Yva))
            self.C, self.d, self.W, self.W0 = C.eval(), d.eval(), W.eval(), W0.eval()  
            
        return self
    
    def predictFull(self, Xnew):
        """Returns predicted values, cadre weights, and cadre estimates for new data"""
        if not self.fitted: print('warning: model not yet fit')
        
        tf.reset_default_graph()
        C  = tf.Variable(self.C, dtype=tf.float32, name='C')
        d  = tf.Variable(self.d, dtype=tf.float32, name='d')
        W  = tf.Variable(self.W, dtype=tf.float32, name='W')
        W0 = tf.Variable(self.W0, dtype=tf.float32, name='w0')
        X = tf.placeholder(dtype=tf.float32, shape=(None,Xnew.shape[1]), name='X')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
                  tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, X)), 
                  tf.abs(d))
    
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                    tf.expand_dims(t,0))), axis=1), T, name='G')                 
    
        ## E[n,y,m] = e^m_y(x^n)
        E = tf.add(tf.einsum('np,ypm->nym', X, W), W0, name='E')
        
        ## F[n,y] = f_y(x^n)
        F = tf.einsum('nm,nym->ny', G, E, name='F')
        Yhat = tf.argmax(F, axis=1)
    
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            Fnew, Lnew, Gnew, mNew = sess.run([F, Yhat, G, bstCd], feed_dict={X: Xnew})
        return Fnew, Lnew, Gnew, mNew
