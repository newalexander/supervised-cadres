## regression.py
## scalar regression

## note that `calcMargiProb`, `calcJointProb`, `calcCondiProb`, and `estEntropy` aren't currently used in any methods.

import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as it
import scipy.special as ss
import time

def eNet(alpha, lam, v):
    """Elastic-net regularization penalty"""
    return lam * (alpha * tf.reduce_sum(tf.abs(v)) + 
                  (1-alpha) * tf.reduce_sum(tf.square(v)))
    
def calcMargiProb(cadId, M):
    """Returns p(M=j) in vector form"""
    return np.array([np.sum(cadId == m) for m in range(M)]) / cadId.shape[0]

def calcJointProb(G, cadId, M):
    """Returns p(M=j, x in C_i) in matrix form"""
    jointProbMat = np.zeros((M,M)) # p(M=j, x in C_i)
    for i,j in it.product(range(M), range(M)):
        jointProbMat[i,j] = np.sum(G[cadId==i,j])
    jointProbMat /= G.shape[0]
    return jointProbMat
    
def calcCondiProb(jointProb, margProb):
    """Returns p(M = j | x in C_i)"""
    return np.divide(jointProb, margProb[:,None], out=np.zeros_like(jointProb), where=margProb[:,None]!=0)

def estEntropy(condProb):
    """Returns estimated entropy for each cadre"""
    return -np.sum(ss.xlogy(condProb, condProb), axis=1) / np.log(2)
    
class regressionCadreModel(object):
    
    def __init__(self, M=2, gamma=10., lambda_d=0.01, lambda_W=0.01,
                 alpha_d=0.9, alpha_W=0.9, Tmax=10000, record=100, 
                 eta=1e-3, Nba=50, eps=1e-3):
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
        self.w0 = 0    # regression biases
        self.C = 0     # cadre centers
        self.d = 0     # cadre assignment weights
        ## data
        self.data = None  # copy of data
        self.features = None # names of features
        self.target = None # name of target
        ## outputs
        self.loss = [] # loss trajectory
        self.mse = []  # MSE trajectory for training data
        self.mseVa = [] # MSE trajectory for validation data
        self.time = [] # optimization times by step
    
    def get_params(self, deep=True):
        return {'M': self.M, 'gamma': self.gamma, 'lambda_d': self.lambda_d, 
                'lambda_W': self.lambda_W, 'alpha_d': self.alpha_d, 
                'alpha_W': self.alpha_W, 'Tmax': self.Tmax, 'record': self.record, 
                'eta': self.eta, 'Nba': self.Nba, 'eps': self.eps}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, Dtr, features, target, Dva=None, inits=dict(), seed=16162, progress=False, store=False):
        np.random.seed(seed)
        """Fit regression cadre model"""
        self.features = features
        self.target = target
        Ntr, P = Dtr.shape[0], features.shape[0] # number of training observations, number of features
        self.fitted = True
        
        if store:
            self.data = Dtr
        
        ## extract values from pd.DataFrames for faster access
        data_mat = Dtr.loc[:,features].values
        target_mat = Dtr.loc[:,[target]].values
        
        if Dva is not None:
            data_mat_va = Dva.loc[:,features].values
            target_mat_va = Dva.loc[:,[target]].values
        
        ############################################
        ## tensorflow parameters and placeholders ##
        ############################################
        tf.reset_default_graph()
    
        ## cadre centers parameter
        if 'C' in inits:
            C = tf.Variable(inits['C'], dtype=tf.float32, name='C')
        else:
            C = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(P,self.M)), 
                            dtype=tf.float32, name='C')
        ## cadre determination weights parameter
        if 'd' in inits:
            d = tf.Variable(inits['d'], dtype=tf.float32, name='d')
        else:
            d = tf.Variable(np.random.uniform(size=(P,)), dtype=tf.float32, name='d')
        ## regression hyperplane weights parameter
        if 'W' in inits:
            W = tf.Variable(inits['W'], dtype=tf.float32, name='W')
        else:
            W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(P,self.M)), 
                            dtype=tf.float32, name='W')
        ## regression hyperplane bias parameter
        if 'w0' in inits:
            w0 = tf.Variable(inits['w0'], dtype=tf.float32, name='w0')
        else:
            w0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='w0')
    
        X = tf.placeholder(dtype=tf.float32, shape=(None,P), name='X')
        Y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='Y')
        eta = tf.placeholder(dtype=tf.float32, shape=(), name='eta')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
                  tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, X)), 
                  d)
    
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                             tf.expand_dims(t,0))), axis=1), T, name='G')                 
    
        ## E[n,m] = e_m(x^n)
        E = tf.add(tf.matmul(X, W), w0, name='E')
        
        ## F[n] = f(x^n) = sum_m g_m(x^n) e_m(x^n)
        F = tf.reduce_sum(G * E, axis=1, keepdims=True)
    
        ## L = 1 / N sum_n sum_m g_m(x^n) * (e_m(x^n) - y_n) ^2
        MSE = tf.reduce_mean( (F - Y)**2 )
        l2_d = self.lambda_d * (1 - self.alpha_d) * tf.reduce_sum(d**2)
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(W**2)
        l1_d = self.lambda_d * self.alpha_d * tf.reduce_sum(tf.abs(d))
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(tf.abs(W))
        l2_C = 1e-7 * tf.reduce_sum(C**2)
        
        loss_smooth = MSE + l2_d + l2_W + l2_C
        optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(loss_smooth)
        loss_full = loss_smooth + l1_d + l1_W
        
        ## proximal gradient steps
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
                ## take gradient step
                sess.run(optimizer, feed_dict={X: data_mat[inds,:],
                                               Y: target_mat[inds,:]})
                ## take proximal step
                sess.run([thresh_d, thresh_W], feed_dict={eta: self.eta})
                # record-keeping        
                if not t % self.record:
                    [l, mse_tr] = sess.run([loss_full, MSE], feed_dict={X: data_mat, Y: target_mat})
                    self.loss.append(l)
                    self.mse.append(mse_tr)
                    if Dva is not None:
                        mse_va = MSE.eval(feed_dict={X: data_mat_va, Y: target_mat_va})
                        self.mseVa.append(mse_va)
                    self.time.append(time.time() - t0)
                    
                    if progress:
                        if len(self.time) and Dva is not None:
                            print(t, self.loss[-1], self.mse[-1], self.mseVa[-1], self.time[-1])
                        elif len(self.time):
                            print(t, self.loss[-1], self.mse[-1], self.time[-1])
                        else:
                            print(t)
                    
            self.C, self.d, self.W, self.w0 = C.eval(), d.eval(), W.eval(), w0.eval()
            self.C = pd.DataFrame(self.C, index=self.features)
            self.d = pd.DataFrame(self.d, index=self.features)
            self.W = pd.DataFrame(self.W, index=self.features)
            
        return self
    
    def predictFull(self, Dnew):
        """Returns predicted values, cadre weights, and cadre estimates for new data"""
        if not self.fitted: print('warning: model not yet fit')
        
        tf.reset_default_graph()
        C  = tf.Variable(self.C.values, dtype=tf.float64, name='C')
        d  = tf.Variable(self.d.values[:,0], dtype=tf.float64, name='d')
        W  = tf.Variable(self.W.values, dtype=tf.float64, name='W')
        w0 = tf.Variable(self.w0, dtype=tf.float64, name='w0')
        X = tf.placeholder(dtype=tf.float64, shape=(None,len(self.features)), name='X')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
                  tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, X)), 
                  d)
    
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                    tf.expand_dims(t,0))), axis=1), T, name='G')                 
    
        ## E[n,m] = e_m(x^n)
        E = tf.add(tf.matmul(X, W), w0, name='E')
        
        ## f[n] = f(x^n)
        F = tf.reduce_sum(G * E, axis=1, name='F') # this won't work if minibatch size 1 is used
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            Fnew, Gnew, mNew = sess.run([F, G, bstCd], feed_dict={X: Dnew[self.features].values})
        return Fnew, Gnew, mNew
    
    def predict(self, Dnew):
        """Returns predicted values for new data"""
        return self.predictFull(Dnew)[0]
    
    def score(self, Dnew):
        """Returns average sum-of-squares for new data"""
        Fnew = self.predict(Dnew)
        return ((Fnew - Dnew[self.target].values)**2).mean()
