## regression.py
## scalar regression

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import itertools as it
import scipy.special as ss

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
        self.cadFts = None        # cadre-assignment feature indices
        self.tarFts = None        # target-prediction feature indices
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
        self.sigma = 0 # prediction noise
        ## data
        self.X = None  # copy of input data
        self.Y = None  # copy of target values
        ## outputs
        self.loss = [] # loss trajectory
    
    def get_params(self, deep=True):
        return {'M': self.M, 'gamma': self.gamma, 'lambda_d': self.lambda_d, 
                'lambda_W': self.lambda_W, 'alpha_d': self.alpha_d, 
                'alpha_W': self.alpha_W, 'Tmax': self.Tmax, 'record': self.record, 
                'eta': self.eta, 'Nba': self.Nba, 'eps': self.eps}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, Xtr, Ytr, cadFts=None, tarFts=None, inits=dict(), seed=16162):
        """Fit regression cadre model"""
        if cadFts is not None:
            self.cadFts = cadFts
        else:
            self.cadFts = np.arange(Xtr.shape[1])
            
        if tarFts is not None:
            self.tarFts = tarFts
        else:
            self.tarFts = np.arange(Xtr.shape[1])
            
        Pcad, Ptar, Ntr = len(self.cadFts), len(self.tarFts), Xtr.shape[0]
        # number of ((cadre-assignment, target-prediction) features, training observations)
        self.fitted = True
        self.X = Xtr
        self.Y = Ytr
        
        ############################################
        ## tensorflow parameters and placeholders ##
        ############################################
        tf.reset_default_graph()
    
        ## cadre centers parameter
        if 'C' in inits:
            C = tf.Variable(inits['C'], dtype=tf.float64, name='C')
        else:
            C = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Pcad,self.M)), 
                            dtype=tf.float64, name='C')
        ## cadre determination weights parameter
        if 'd' in inits:
            d = tf.Variable(inits['d'], dtype=tf.float64, name='d')
        else:
            d = tf.Variable(np.random.uniform(size=(Pcad)), dtype=tf.float64, name='d')
        ## regression hyperplane weights parameter
        if 'W' in inits:
            W = tf.Variable(inits['W'], dtype=tf.float64, name='W')
        else:
            W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Ptar,self.M)), 
                            dtype=tf.float64, name='W')
        ## regression hyperplane bias parameter
        if 'w0' in inits:
            w0 = tf.Variable(inits['w0'], dtype=tf.float64, name='w0')
        else:
            w0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float64), 
                             dtype=tf.float64, name='w0')
        ## model error parameter
        if 'sigma' in inits:
            sigma = tf.Variable(inits['sigma'], dtype=tf.float64, name='sigma')
        else:
            sigma = tf.Variable(0.1, dtype=tf.float64, name='sigma')
    
        Xcad = tf.placeholder(dtype=tf.float64, shape=(None,Pcad), name='Xcad')
        Xtar = tf.placeholder(dtype=tf.float64, shape=(None,Ptar), name='Xtar')
        N = tf.cast(tf.gather(tf.shape(Xcad), 0), dtype=tf.float64, name='N')
        Y = tf.placeholder(dtype=tf.float64, shape=(None,1), name='Y')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
                  tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, Xcad)), 
                  tf.abs(d))
    
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                             tf.expand_dims(t,0))), axis=1), T, name='G')                 
    
        ## E[n,m] = e_m(x^n)
        E = tf.add(tf.matmul(Xtar, W), w0, name='E')
    
        ## L = 1 / N sum_n sum_m g_m(x^n) * (e_m(x^n) - y_n) ^2
        L = tf.add(tf.reduce_mean(tf.reduce_sum(G * (E - Y) **2, axis=1)) / 2 / sigma**2,
             (eNet(self.alpha_d, self.lambda_d, d) + 
              eNet(self.alpha_W, self.lambda_W, W)) / 2 / N / sigma**2 +
             (1/2 + 1/N) * tf.log(sigma**2), name='L')
    
        optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(L)
        
        ####################
        ## learning model ##
        ####################
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            
            ## perform optimization
            for t in range(self.Tmax):
                inds = np.random.choice(Ntr, self.Nba, replace=False)      
                sess.run(optimizer, feed_dict={Xcad: Xtr[np.ix_(inds, self.cadFts)],
                                               Xtar: Xtr[np.ix_(inds, self.tarFts)],
                                               Y: Ytr[inds]})
                # record-keeping        
                if not t % self.record:
                    self.loss.append(L.eval(feed_dict={
                            Xcad: Xtr[:,self.cadFts],
                            Xtar: Xtr[:,self.tarFts],
                            Y: Ytr}))
                    if len(self.loss) > 2 and (np.abs(self.loss[-1] - self.loss[-2]) < self.eps):
                        break   
            self.C, self.d, self.W, self.w0, self.sigma = C.eval(), d.eval(), W.eval(), w0.eval(), sigma.eval()    
            
        return self
    
    def predictFull(self, Xnew):
        """Returns predicted values, cadre weights, and cadre estimates for new data"""
        if not self.fitted: print('warning: model not yet fit')
        
        tf.reset_default_graph()
        C  = tf.Variable(self.C, dtype=tf.float64, name='C')
        d  = tf.Variable(self.d, dtype=tf.float64, name='d')
        W  = tf.Variable(self.W, dtype=tf.float64, name='W')
        w0 = tf.Variable(self.w0, dtype=tf.float64, name='w0')
        Xcad = tf.placeholder(dtype=tf.float64, shape=(None,len(self.cadFts)), name='X')
        Xtar = tf.placeholder(dtype=tf.float64, shape=(None,len(self.tarFts)), name='X')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
                  tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, Xcad)), 
                  tf.abs(d))
    
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                    tf.expand_dims(t,0))), axis=1), T, name='G')                 
    
        ## E[n,m] = e_m(x^n)
        E = tf.add(tf.matmul(Xtar, W), w0, name='E')
        
        ## f[n] = f(x^n)
        F = tf.reduce_sum(G * E, axis=1, name='F') # this won't work if minibatch size 1 is used
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            Fnew, Gnew, mNew = sess.run([F, G, bstCd], feed_dict={Xcad: Xnew[:,self.cadFts],
                                                                  Xtar: Xnew[:,self.tarFts]})
        return Fnew, Gnew, mNew
    
        
    def predict(self, Xnew):
        """Returns predicted values for new data"""
        return self.predictFull(Xnew)[0]
    
    def score(self, Xnew, Ynew):
        """Returns average sum-of-squares for new data"""
        Fnew = self.predict(Xnew)
        return ((Fnew - np.squeeze(Ynew))**2).mean()
    
    def entropy(self, Xnew):
        """Returns estimated entropy for each cadre"""
        G, m = self.predictFull(Xnew)[1:]    
        marg = calcMargiProb(m, self.M)
        jont = calcJointProb(G, m,  self.M)
        cond = calcCondiProb(jont, marg)
        return estEntropy(cond)
    
    def getNumberParams(self):
        """Returns number of parameters of a model"""
        return np.prod(self.C.shape) + np.prod(self.d.shape) + np.prod(self.W.shape) + np.prod(self.w0.shape)

    def getNumberParamsRed(self, threshold=1e-4):
        """Returns number of active parameters of a model"""
        if not self.fitted: print('warning: model not yet fit')
        return (np.sum(np.abs(self.C) > threshold) + np.sum(np.abs(self.d) > threshold) + 
                np.sum(np.abs(self.W) > threshold) + np.sum(np.abs(self.w0 > threshold)))
    
    def calcBIC(self):
        """Returns BIC of learned model"""
        if not self.fitted: print('warning: model not yet fit')
        return 2*self.Y.shape[0]*self.loss[-1] + 2 * self.getNumberParams() * np.log(self.Y.shape[0])

    def calcBICred(self, threshold=1e-4):
        """Returns effective degree-of-freedom BIC of learned model"""
        if not self.fitted: print('warning: model not yet fit')
        return 2*self.Y.shape[0]*self.loss[-1] + 2 * self.getNumberParamsRed(threshold) * np.log(self.Y.shape[0])
