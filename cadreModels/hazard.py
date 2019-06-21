## hazardCadreRisk.py
## precision proportional hazards model written to allow separate specification
## of cadre-assignment and target-prediction features
## data input assumed to be a pandas.DataFrame

from __future__ import division, print_function, absolute_import

import time
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from itertools import product
from scipy.special import xlogy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from lifelines.utils import concordance_index

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
    for i,j in product(range(M), range(M)):
        jointProbMat[i,j] = np.sum(G[cadId==i,j])
    jointProbMat /= G.shape[0]
    return jointProbMat
    
def calcCondiProb(jointProb, margProb):
    """Returns p(M = j | x in C_i)"""
    return np.divide(jointProb, margProb[:,None], out=np.zeros_like(jointProb), where=margProb[:,None]!=0)

def estEntropy(condProb):
    """Returns estimated entropy for each cadre"""
    return -np.sum(xlogy(condProb, condProb), axis=1) / np.log(2)

class hazardCadreModel(object):
    
    def __init__(self, M=2, gamma=10., lambda_d=0.01, lambda_W=0.01,
                 alpha_d=0.9, alpha_W=0.9, Kmax=10000, record=100, 
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
        self.Kmax = Kmax     # maximum iterations
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
        self.data = None      # copy of input data
        self.cadFts = None    # cadre-assignment features
        self.tarFts = None    # target-prediction features
        self.columns = None   # all column-names
        self.timeCol = None   # time to event column name
        self.statusCol = None # censor status column name
        ## outputs
        self.loss = [] # loss trajectory
        self.time = [] # times
        self.grad = {'C': [], 'd': [], 'W': []}
        self.norm = {'C': [], 'd': [], 'W': []}
        self.conc = [] # training set concordances
        self.ccVa = [] # validation set concordances
        self.vars = [] # cadre-membership weight variance
    
    def get_params(self, deep=True):
        return {'M': self.M, 'gamma': self.gamma, 'lambda_d': self.lambda_d, 
                'lambda_W': self.lambda_W, 'alpha_d': self.alpha_d, 
                'alpha_W': self.alpha_W, 'Kmax': self.Kmax, 'record': self.record, 
                'eta': self.eta, 'Nba': self.Nba, 'eps': self.eps}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, data, timeCol, statusCol, cadFts=None, tarFts=None, dataVa=None, inits=dict(), seed=16162, store=False):
        np.random.seed(seed)
        """Fits proportional hazards model"""
        ## store categories of column names
        if cadFts is not None:
            self.cadFts = cadFts
        else:
            self.cadFts = data.columns
        if tarFts is not None:
            self.tarFts = tarFts
        else:
            self.tarFts = data.columns
        self.columns = data.columns
        self.timeCol = timeCol
        self.statusCol = statusCol
        ## get dataset attributes
        Pcad, Ptar, Ntr = cadFts.shape[0], tarFts.shape[0], data.shape[0]
        if dataVa is not None:
            Nva = dataVa.shape[0]
        self.fitted = True
        if store:
            self.data = data
            
        ## separate into different arrays for faster access
        dataCad = data.loc[:,self.cadFts].values
        dataTar = data.loc[:,self.tarFts].values
        
        dataTime = data.loc[:,self.timeCol].values
        dataStatus = data.loc[:,self.statusCol].values
        
        if dataVa is not None:
            dataCadVa = dataVa.loc[:,self.cadFts].values
            dataTarVa = dataVa.loc[:,self.tarFts].values
        
            dataTimeVa = dataVa.loc[:,self.timeCol].values
            dataStatusVa = dataVa.loc[:,self.statusCol].values
        
        ############################################
        ## tensorflow parameters and placeholders ##
        ############################################
        tf.reset_default_graph()
    
        ## cadre centers parameter
        if 'C' in inits:
            C = tf.Variable(inits['C'], dtype=tf.float32, name='C')
        else:
            C = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Pcad,self.M)), 
                            dtype=tf.float32, name='C')
        ## cadre determination weights parameter
        if 'd' in inits:
            d = tf.Variable(inits['d'], dtype=tf.float32, name='d')
        else:
            d = tf.Variable(np.random.uniform(size=(Pcad)), dtype=tf.float32, name='d')
        ## regression hyperplane weights parameter
        if 'W' in inits:
            W = tf.Variable(inits['W'], dtype=tf.float32, name='W')
        else:
            W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Ptar,self.M)), 
                            dtype=tf.float32, name='W')
        ## regression hyperplane bias parameter
        if 'W0' in inits:
            W0 = tf.Variable(inits['W0'], dtype=tf.float32, name='w0')
        else:
            W0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='w0')
    
        Xcad = tf.placeholder(dtype=tf.float32, shape=(None,Pcad), name='Xcad')
        Xtar = tf.placeholder(dtype=tf.float32, shape=(None,Ptar), name='Xtar')
        R = tf.placeholder(dtype=tf.float32, shape=(None, None), name='R')
        E = tf.placeholder(dtype=tf.int32, shape=(None,), name='E')
        eta = tf.placeholder(dtype=tf.float32, shape=(), name='eta')
        
        ## T[n,m] = ||x^n - c^m||^2_D (weighted distances)
        T = tf.einsum('npm,p->nm', 
                  tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, Xcad)), 
                  tf.abs(d))
                   
        ## G[n,m] = g_m(x^n) (cadre membership probabilities)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                             tf.expand_dims(t,0))), axis=1), T, name='G')                 

        ## S[n,m] = s_m(x^n) (cadre-wise hazard scores)
        S = tf.exp(tf.matmul(Xtar, W) + W0, name='S')
        
        ## F[n,k] = f_k(x^n) (combined hazard scores)
        F = tf.log(tf.reduce_sum(G * S, axis=1, name='F', keepdims=True))
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        H = tf.exp(F)

        H_tile = tf.transpose(tf.tile(H, (1, self.Nba)), perm=(1,0))
        gatherScore = tf.gather(F, E)
        gatherTile  = tf.gather(R * H_tile, E)
        
        ## error and regularization terms
        lambda_G = 1e-4
        score = -tf.reduce_mean(gatherScore)  
        partition = tf.reduce_mean(tf.log(tf.reduce_sum(gatherTile, axis=1)))
        l2_d = self.lambda_d * (1 - self.alpha_d) * tf.reduce_sum(d**2)
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(W**2)
        l1_d = self.lambda_d * self.alpha_d * tf.reduce_sum(tf.abs(d))
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(tf.abs(W))
        l2_C = 1e-6 * tf.reduce_sum(C**2)
        reg_G = -lambda_G * tf.reduce_mean(tf.nn.moments(G, axes=1)[1])
        
        ## smooth loss function
        L = score + partition + l2_d + l2_W + l2_C #+ reg_G
        opt = tf.train.AdamOptimizer(learning_rate=eta)
        optimizer = opt.minimize(L)#, var_list=[W])
        
        ## nonsmooth proximal terms
        thresh_W = tf.assign(W, tf.sign(W) * (tf.abs(W) - eta * self.lambda_W * self.alpha_W) * tf.cast(tf.abs(W) > eta * self.lambda_W * self.alpha_W, tf.float32))
        thresh_d = tf.assign(d, tf.maximum(0., tf.sign(d) * (tf.abs(d) - eta * self.lambda_d * self.alpha_d) * tf.cast(tf.abs(d) > eta * self.lambda_d * self.alpha_d, tf.float32)))
        Lfull = L + l1_d + l1_W
        
        grad = tf.gradients(L, [d, C, W])
        
        ####################
        ## learning model ##
        ####################
        t0 = time.time()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()            
            
            t0 = time.time()
            ## perform optimization
            for k in range(self.Kmax):
                ## preprocess minibatch
                inds = np.random.choice(Ntr, self.Nba, replace=False)
                Rsets = (dataTime[None,inds] >= dataTime[inds,None]).astype(np.float32)
                Esets = np.where(dataStatus[inds])[0]
                
#                 print('before optimization', time.time() - t0)
                sess.run(optimizer, feed_dict={Xcad: dataCad[inds,:], Xtar: dataTar[inds,:], R: Rsets, E: Esets, eta: self.eta})
#                 print('after optimization/before thresholding', time.time() - t0)
                sess.run([thresh_d, thresh_W], feed_dict={eta: self.eta})
#                 print('after thresholding/before recording', time.time() - t0)
                
                # record-keeping        
                if not k % self.record:
                    [l, gradd, gradC, gradW] = sess.run(
                            [Lfull]+grad, feed_dict={Xcad: dataCad[inds,:], Xtar: dataTar[inds,:], R: Rsets, E: Esets, eta: self.eta})
                    de, Ce, We = sess.run([d, C, W])
                    self.loss.append(l)
                    self.grad['d'].append(np.linalg.norm(gradd))
                    self.grad['C'].append(np.linalg.norm(gradC))
                    self.grad['W'].append(np.linalg.norm(gradW))
                    self.norm['d'].append(np.linalg.norm(de))
                    self.norm['C'].append(np.linalg.norm(Ce))
                    self.norm['W'].append(np.linalg.norm(We))
                    ## calculate training set concordance
                    inds = np.random.choice(Ntr, np.minimum(Ntr, 10000), replace=False)
                    haz = H.eval(feed_dict={Xcad: dataCad[inds,:], Xtar: dataTar[inds,:]})
                    self.conc.append(concordance_index(dataTime[inds], -haz, dataStatus[inds]))
                    ## calculate validation set concordance
                    if dataVa is not None:
                        inds = np.random.choice(Nva, np.minimum(Nva, 10000), replace=False)
                        haz = H.eval(feed_dict={Xcad: dataCadVa[inds,:], Xtar: dataTarVa[inds,:]})
                        self.ccVa.append(concordance_index(dataTimeVa[inds], -haz, dataStatusVa[inds]))
                    ## calculate training set membership weight variance
                    inds = np.random.choice(Ntr, np.minimum(Ntr, 1000), replace=False)
                    self.vars.append(reg_G.eval(feed_dict={Xcad: dataCad[inds,:], Xtar: dataTar[inds,:]}) / -lambda_G)
                    self.time.append(time.time() - t0)
                    if not k % 200:
                        print(k, self.loss[-1], self.conc[-1], self.vars[-1], self.time[-1])
            self.C, self.d, self.W, self.W0 = C.eval(), d.eval(), W.eval(), W0.eval()
            self.C = pd.DataFrame(self.C, index=cadFts)
            self.d = pd.DataFrame(self.d, index=cadFts)
            self.W = pd.DataFrame(self.W, index=tarFts)
            
        return self
    
    def predictFull(self, Xnew):
        """Returns predicted values, cadre weights, and cadre estimates for new data"""
        if not self.fitted: print('warning: model not yet fit')
        Pcad, Ptar = self.cadFts.shape[0], self.tarFts.shape[0]
        
        tf.reset_default_graph()
        C  = tf.Variable(self.C.values, dtype=tf.float32, name='C')
        d  = tf.Variable(self.d.values[:,0], dtype=tf.float32, name='d')
        W  = tf.Variable(self.W.values, dtype=tf.float32, name='W')
        W0 = tf.Variable(self.W0, dtype=tf.float32, name='w0')
        Xcad = tf.placeholder(dtype=tf.float32, shape=(None,Pcad), name='Xcad')
        Xtar = tf.placeholder(dtype=tf.float32, shape=(None,Ptar), name='Xtar')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
                  tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, Xcad)), 
                  tf.abs(d))
    
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                    tf.expand_dims(t,0))), axis=1), T, name='G')                 
    
        ## E[n,y,m] = e^m_y(x^n)
        S = tf.exp(tf.matmul(Xtar, W) + W0, name='S')
        
        ## F[n,y] = f_y(x^n)
        F = tf.log(tf.reduce_sum(G * S, axis=1, name='F'))
    
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            Fnew, Tnew, Gnew, mNew = sess.run([F, T, G, bstCd], feed_dict={Xcad: Xnew[self.cadFts].values,
                                                                          Xtar: Xnew[self.tarFts].values})
        return Fnew, Tnew, Gnew, mNew
   
    def predict(self, Xnew):
        """Returns predicted values for new data"""
        return self.predictFull(Xnew)[0]
   
    def score(self, Xnew):
        """Returns concordance for new data"""
        Fnew = self.predict(Xnew)
        return concordance_index(Xnew.loc[:,self.timeCol], -Fnew, Xnew.loc[:,self.statusCol])
    
    def getNumberParams(self):
        """Returns number of parameters of a model"""
        return np.prod(self.C.shape) + np.prod(self.d.shape) + np.prod(self.W.shape) + np.prod(self.W0.shape)
    
    def calcBIC(self, N):
        """Returns BIC of learned model"""
        if not self.fitted: print('warning: model not yet fit')
        return 2*N*self.loss[-1] + 2 * self.getNumberParams()*np.log(N)
    
    def entropy(self, Xnew):
        """Returns estimated entropy for each cadre"""
        __, __, G, m = self.predictFull(Xnew)    
        marg = calcMargiProb(m, self.M)
        jont = calcJointProb(G, m, self.M)
        cond = calcCondiProb(jont, marg)
        return estEntropy(cond)
    
    def calcLoss(self, Xnew):
        """Returns loss function value evaluated on all of Xnew"""
        dataCad = Xnew.loc[:,self.cadFts].values
        dataTar = Xnew.loc[:,self.tarFts].values
        
        dataTime = Xnew.loc[:,self.timeCol].values
        dataStatus = Xnew.loc[:,self.statusCol].values
        
        N = np.minimum(Xnew.shape[0], 45000)
        
        tf.reset_default_graph()
        if not self.fitted: print('warning: model not yet fit')
        Pcad, Ptar = self.cadFts.shape[0], self.tarFts.shape[0]
        
        tf.reset_default_graph()
        C  = tf.Variable(self.C.values, dtype=tf.float32, name='C')
        d  = tf.Variable(self.d.values[:,0], dtype=tf.float32, name='d')
        W  = tf.Variable(self.W.values, dtype=tf.float32, name='W')
        W0 = tf.Variable(self.W0, dtype=tf.float32, name='w0')
        Xcad = tf.placeholder(dtype=tf.float32, shape=(None,Pcad), name='Xcad')
        Xtar = tf.placeholder(dtype=tf.float32, shape=(None,Ptar), name='Xtar')
        R = tf.placeholder(dtype=tf.float32, shape=(None, None), name='R')
        E = tf.placeholder(dtype=tf.int32, shape=(None,), name='E')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
                  tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, Xcad)), 
                  tf.abs(d))
    
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                    tf.expand_dims(t,0))), axis=1), T, name='G')                 
    
        ## E[n,y,m] = e^m_y(x^n)
        S = tf.exp(tf.matmul(Xtar, W) + W0, name='S')
        
        ## F[n,y] = f_y(x^n)
        F = tf.log(tf.reduce_sum(G * S, axis=1, name='F', keepdims=True))       
        H = tf.exp(F)

        H_tile = tf.transpose(tf.tile(H, (1, N)), perm=(1,0))
        gatherScore = tf.gather(F, E)
        gatherTile  = tf.gather(R * H_tile, E)
        
        score = -tf.reduce_mean(gatherScore)  
        partition = tf.reduce_mean(tf.log(tf.reduce_sum(gatherTile, axis=1)))
        l2_d = self.lambda_d * (1 - self.alpha_d) * tf.reduce_sum(d**2)
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(W**2)
        l1_d = self.lambda_d * self.alpha_d * tf.reduce_sum(tf.abs(d))
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(tf.abs(W))
        l2_C = 1e-6 * tf.reduce_sum(C**2)
        
        ## smooth loss function
        L = score + partition + l2_d + l2_W + l2_C
        Lfull = L + l1_d + l1_W
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            np.random.seed(12515)
            inds = np.random.choice(Xnew.shape[0], N, replace=False)
            
            Rsets = (dataTime[None,inds] >= dataTime[inds,None]).astype(np.float32)
            Esets = np.where(dataStatus[inds])[0]
            
            l = Lfull.eval(feed_dict={Xcad: dataCad[inds,:], Xtar: dataTar[inds,:], R: Rsets, E: Esets})
        return l
