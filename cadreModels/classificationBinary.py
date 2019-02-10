## classificationBinary.py
## binary classification cadres with hinge-loss

from __future__ import division, print_function, absolute_import

import time
import numpy as np
import tensorflow as tf

def eNet(alpha, lam, v):
    """Elastic-net regularization penalty"""
    return lam * (alpha * tf.reduce_sum(tf.abs(v)) + 
                  (1-alpha) * tf.reduce_sum(tf.square(v)))
    
class binaryCadreModel(object):
    
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
        
    def fit(self, Xtr, Ytr, Xva=None, Yva=None, names=None, seed=16162, store=False, progress=False):
        np.random.seed(seed)
        """Fits binary classification cadre model"""
        if names is not None:
            self.columns = names
            
        Ntr, P = Xtr.shape
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
        W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(P,self.M)), 
                            dtype=tf.float32, name='W')
        ## regression hyperplane bias parameter
        W0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='W0')
    
        X = tf.placeholder(dtype=tf.float32, shape=(None,P), name='X')
        Y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='Y')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
              tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, X)), 
              d)
                
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        ## cadre-assignment scores
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                             tf.expand_dims(t,0))), axis=1), T, name='G')                 

        ## E[n,y,m] = e^m_y(x^n)
        ## cadre-wise prediction scores
        E = tf.add(tf.matmul(X, W), W0, name='E')
        
        ## F[n] = f_k(x^n)
        F = tf.reduce_sum(G * E, name='F', axis=1, keepdims=True)
        
        ## predictd label
        Yhat = tf.sign(F)
        ## classification rate
        class_rate = tf.reduce_mean(tf.cast(tf.equal(Yhat, Y), tf.float64), name='rate')

        ## L = 1 / N sum_n log(p(y[n] | x[n])) + reg(Theta)
        hinge_loss = tf.reduce_mean(tf.nn.relu(1 - F * Y))
        l2_d = self.lambda_d * (1 - self.alpha_d) * tf.reduce_sum(d**2)
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(W**2)
        l1_d = self.lambda_d * self.alpha_d * tf.reduce_sum(tf.abs(d))
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(tf.abs(W))
        l2_C = 1e-7 * tf.reduce_sum(C**2)
        
        ## loss that is fed into optimizer
        loss_opt = hinge_loss + l2_d + l2_W + l1_d + l1_W + l2_C
        ## full loss, including l1 terms handled with proximal gradient
        loss_full = loss_opt + l1_d + l1_W
        optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(loss_opt)
        
        ## nonsmooth proximal terms
        thresh_W = tf.assign(W, tf.sign(W) * (tf.abs(W) - eta * self.lambda_W * self.alpha_W) * tf.cast(tf.abs(W) > eta * self.lambda_W * self.alpha_W, tf.float32))
        thresh_d = tf.assign(d, tf.maximum(0., tf.sign(d) * (tf.abs(d) - eta * self.lambda_d * self.alpha_d) * tf.cast(tf.abs(d) > eta * self.lambda_d * self.alpha_d, tf.float32)))
        
        ####################
        ## learning model ##
        ####################
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            t0 = time.time()
            ## perform optimization
            for t in range(self.Tmax):
                inds = np.random.choice(Ntr, self.Nba, replace=False)
                sess.run(optimizer, feed_dict={X: Xtr[inds,:], Y: Ytr[inds,:]})
                sess.run([thresh_d, thresh_W], feed_dict={eta: self.eta})
                # record-keeping        
                if not t % self.record:
                    if progress:
                        if len(self.time) and Xva is not None:
                            print(t, self.loss[-1], self.accs[-1], self.accsVa[-1], self.time[-1])
                        elif len(self.time):
                            print(t, self.loss[-1], self.accs[-1], self.time[-1])
                        else:
                            print(t)
                    self.time.append(time.time() - t0)
                    l, rate = sess.run([loss_full, class_rate], feed_dict={X: Xtr, Y: Ytr})
                    self.loss.append(l)
                    self.accs.append(rate)
                    if Xva is not None:
                        rate = class_rate.eval(feed_dict={X: Xva, Y: Yva})
                        self.accsVa.append(rate)
            self.C, self.d, self.W, self.W0 = C.eval(), d.eval(), W.eval(), W0.eval()  
            
        return self
    
    def predictFull(self, Xnew):
        """Returns classification scores, predicted labels, cadre membership scores, and predicted cadres"""
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
        ## cadre-wise prediction scores
        E = tf.add(tf.matmul(X, W), W0, name='E')
        
        ## F[n] = f_k(x^n)
        F = tf.reduce_sum(G * E, name='F', axis=1, keepdims=True)
        
        ## predicted label
        Yhat = tf.sign(F)
        ## predicted cadre
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            Fnew, Lnew, Gnew, mNew = sess.run([F, Yhat, G, bstCd], feed_dict={X: Xnew})
        return Fnew, Lnew, Gnew, mNew
   
    def predictClass(self, Xnew):
        __, Lnew, __, __ = self.predictFull(Xnew)
        return Lnew
    
    def score(self, Xnew, Ynew):
        __, Lnew, __, __ = self.predictFull(Xnew)
        return np.mean(Ynew == Lnew)
