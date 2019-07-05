## classificationBinary.py
## binary classification cadres with logistic loss
## parameters have both l1 and l2 regularization, and l1 terms are handled with
## proximal gradient steps
## input is a pandas.DataFrame containing both target and input features
## loss function uses jensen's inequality for better-conditioned gradients

## to-do: need to add methods for cadre assessment metrics (entropy, weight variance, sparsity, average best match?)

from __future__ import division, print_function, absolute_import

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import utility as u

from itertools import product
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import xlogy

class binaryCadreModel(object):
    
    def __init__(self, M=2, gamma=10., lambda_d=0.01, lambda_W=0.01,
                 alpha_d=0.9, alpha_W=0.9, Tmax=10000, record=100, 
                 eta=2e-3, Nba=64, eps=1e-3, termination_metric='ROC_AUC'):
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
        self.termination_metric = termination_metric
        ## parameters
        self.W = 0     # regression weights
        self.W0 = 0    # regression biases
        self.C = 0     # cadre centers
        self.d = 0     # cadre assignment weights
        ## data
        self.data = None       # pd.DataFrame containing features and response
        self.cadreFts = None   # pd.Index of column-names giving features used for cadre assignment
        self.predictFts = None # pd.Index of column-names giving features used for target-prediction
        self.targetCol = None  # string column-name of response variable
        ## outputs
        self.metrics = {'training': {'loss': [],
                                     'accuracy': [],
                                     'ROC_AUC': [],
                                     'PR_AUC': []},
                        'validation': {'loss': [],
                                      'accuracy': [],
                                      'ROC_AUC': [],
                                      'PR_AUC': []}}
        self.time = [] # times
        self.proportions = [] # cadre membership proportions during training
        self.termination_reason = None # why training stopped
    
    def get_params(self, deep=True):
        return {'M': self.M, 'gamma': self.gamma, 'lambda_d': self.lambda_d, 
                'lambda_W': self.lambda_W, 'alpha_d': self.alpha_d, 
                'alpha_W': self.alpha_W, 'Tmax': self.Tmax, 'record': self.record, 
                'eta': self.eta, 'Nba': self.Nba, 'eps': self.eps}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, data, targetCol, cadreFts=None, predictFts=None, dataVa=None, 
            seed=16162, store=False, progress=False, inits=None):
        np.random.seed(seed)
        """Fits binary classification cadre model"""
        ## store categories of column names
        self.targetCol = targetCol
        if cadreFts is not None:
            self.cadreFts = cadreFts
        else:
            self.cadreFts = data.drop(targetCol, axis=1).columns
        if predictFts is not None:
            self.predictFts = predictFts
        else:
            self.predictFts = data.drop(targetCol, axis=1).columns
        ## get dataset attributes
        self.fitted = True
        if store:
            self.data = data
        Pcadre, Ppredict, Ntr = self.cadreFts.shape[0], self.predictFts.shape[0], data.shape[0]
            
        ## split data into separate numpy arrays for faster access
        ## features for cadre-assignment
        dataCadre = data.loc[:,self.cadreFts].values
        ## features for target-prediction
        dataPredict = data.loc[:,self.predictFts].values
        ## target feature
        dataTarget = data.loc[:,[self.targetCol]].values
        target_tr = 2 * dataTarget - 1
        
        if dataVa is not None:
            dataCadreVa = dataVa.loc[:,self.cadreFts].values
            dataPredictVa = dataVa.loc[:,self.predictFts].values
            dataTargetVa = dataVa.loc[:,[self.targetCol]].values
            target_va = 2 * dataTargetVa - 1
                
        ############################################
        ## tensorflow parameters and placeholders ##
        ############################################
        tf.reset_default_graph()
    
        ## cadre centers parameter
        if inits is not None and 'C' in inits:
            C = tf.Variable(inits['C'], dtype=tf.float32, name='C')
        else:
            C = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Pcadre,self.M)), 
                            dtype=tf.float32, name='C')
        ## cadre determination weights parameter
        if inits is not None and 'd' in inits:
            d = tf.Variable(inits['d'], dtype=tf.float32, name='d')
        else:
            d = tf.Variable(np.random.uniform(size=(Pcadre)), dtype=tf.float32, name='d')
        
        ## regression hyperplane weights parameter
        if inits is not None and 'W' in inits:
            W = tf.Variable(inits['W'], dtype=tf.float32, name='W')
        else:
            W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Ppredict,self.M)), 
                            dtype=tf.float32, name='W')
        ## regression hyperplane bias parameter
        if inits is not None and 'W0' in inits:
            W0 = tf.Variable(inits['W0'], dtype=tf.float32, name='W0')
        else:
            W0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='W0')
    
        Xcadre = tf.placeholder(dtype=tf.float32, shape=(None,Pcadre), name='Xcadre')
        Xpredict = tf.placeholder(dtype=tf.float32, shape=(None,Ppredict), name='Xpredict')
        Y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='Y')
        eta = tf.placeholder(dtype=tf.float32, shape=(), name='eta')
        lambda_Ws = tf.placeholder(dtype=tf.float32, shape=(self.M,), name='lambda_Ws')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
              tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, Xcadre)), 
              d)
                
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        ## cadre-assignment scores
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                             tf.expand_dims(t,0))), axis=1), T, name='G')                 

        ## E[n,y,m] = e^m_y(x^n)
        ## cadre-wise prediction scores
        E = tf.add(tf.matmul(Xpredict, W), W0, name='E')
        
        ## F[n] = f_k(x^n)
        F = tf.reduce_sum(G * E, name='F', axis=1, keepdims=True)
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        ## observation-wise error terms (based on jensen's inequality)
        error_terms = tf.log(1 + tf.reduce_sum(G * tf.exp(-Y * E), axis=1))
        loss_score = tf.reduce_mean(error_terms)
        
        ## regularization
        l2_d = self.lambda_d * (1 - self.alpha_d) * tf.reduce_sum(d**2)
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(lambda_Ws * W**2)
        l1_d = self.lambda_d * self.alpha_d * tf.reduce_sum(tf.abs(d))
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(lambda_Ws * tf.abs(W))
        l2_C = 1e-7 * tf.reduce_sum(C**2)
        
        ## loss that is fed into optimizer
        loss_opt = loss_score + l2_d + l2_W + l2_C
        ## full loss, including l1 terms handled with proximal gradient
        loss_full = loss_opt + l1_d + l1_W
        optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_opt)
        
        ## nonsmooth proximal terms
        thresh_W = tf.assign(W, tf.sign(W) * (tf.abs(W) - eta * self.lambda_W * lambda_Ws * self.alpha_W) * tf.cast(tf.abs(W) > eta * self.lambda_W * self.alpha_W, tf.float32))
        thresh_d = tf.assign(d, tf.maximum(0., tf.sign(d) * (tf.abs(d) - eta * self.lambda_d * self.alpha_d) * tf.cast(tf.abs(d) > eta * self.lambda_d * self.alpha_d, tf.float32)))
        
        ####################
        ## learning model ##
        ####################
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            
            if progress:
                if dataVa is not None:
                    print('numbers being printed:', 
                          'SGD iteration, training loss, training accuracy, validation loss, validation accuracy, time')
                else:
                    print('numbers being printed:',
                          'SGD iteration, training loss, training accuracy, time')

            t0 = time.time()
            ## perform optimization
            for t in range(self.Tmax):
                inds = np.random.choice(Ntr, self.Nba, replace=False)
                ## calculate adaptive regularization parameter
                cadres = bstCd.eval(feed_dict={Xcadre: dataCadre[inds,:], Xpredict: dataPredict[inds,:]})
                cadre_counts = np.zeros(self.M)
                for m in range(self.M):
                    cadre_counts[m] = np.sum(cadres == m) + 1
                cadre_counts = cadre_counts.sum() / cadre_counts
                
                ## take SGD step
                sess.run(optimizer, feed_dict={Xcadre: dataCadre[inds,:],
                                               Xpredict: dataPredict[inds,:],
                                               Y: target_tr[inds,:],
                                               lambda_Ws: cadre_counts,
                                               eta: self.eta / np.sqrt(t+1)})
                ## take proximal gradient step
                sess.run([thresh_d, thresh_W], feed_dict={eta: self.eta / np.sqrt(t+1), lambda_Ws: cadre_counts})
                # record-keeping        
                if not t % self.record:
                    if progress:
                        if len(self.time) and dataVa is not None:
                            print(t,
                                  self.metrics['training']['loss'][-1], 
                                  self.metrics['training']['accuracy'][-1],
                                  self.metrics['validation']['loss'][-1], 
                                  self.metrics['validation']['accuracy'][-1],
                                  self.time[-1])
                        elif len(self.time):
                            print(t,
                                  self.metrics['training']['loss'][-1], 
                                  self.metrics['training']['accuracy'][-1],
                                  self.time[-1])
                        else:
                            print(t)
                    self.time.append(time.time() - t0)
                    ## calculate metrics -- this should be its own function since it gets repeated
                    cadres = bstCd.eval(feed_dict={Xcadre: dataCadre, Xpredict: dataPredict})
                    cadre_counts = np.zeros(self.M)
                    for m in range(self.M):
                        cadre_counts[m] = np.sum(cadres == m) + 1
                    cadre_counts = cadre_counts / cadre_counts.sum()
                    l, margin, cadres = sess.run([loss_full, F, bstCd], 
                                                 feed_dict={Xcadre: dataCadre,
                                                            Xpredict: dataPredict,
                                                            lambda_Ws: cadre_counts,
                                                            Y: target_tr})
                    yhat = 0.5 * (np.sign(margin) + 1)
                    self.metrics['training']['loss'].append(l)
                    self.metrics['training']['accuracy'].append(np.mean(yhat == dataTarget))
                    self.metrics['training']['ROC_AUC'].append(roc_auc_score(dataTarget,
                                                                             margin))
                    self.metrics['training']['PR_AUC'].append(average_precision_score(dataTarget,
                                                                                      margin))
                    self.proportions.append(pd.Series(cadres).value_counts().T)
                    self.proportions[-1] /= self.proportions[-1].sum()
                        
                    if dataVa is not None:
                        cadres = bstCd.eval(feed_dict={Xcadre: dataCadreVa, Xpredict: dataPredictVa})
                        cadre_counts = np.zeros(self.M)
                        for m in range(self.M):
                            cadre_counts[m] = np.sum(cadres == m) + 1
                        cadre_counts = cadre_counts / cadre_counts.sum()
                        l, margin = sess.run([loss_full, F], feed_dict={Xcadre: dataCadreVa,
                                                                        Xpredict: dataPredictVa,
                                                                        lambda_Ws: cadre_counts,
                                                                        Y: target_va})
                        yhat = 0.5 * (np.sign(margin) + 1)
                        self.metrics['validation']['loss'].append(l)
                        self.metrics['validation']['accuracy'].append(np.mean(yhat == dataTargetVa))
                        self.metrics['validation']['ROC_AUC'].append(roc_auc_score(dataTargetVa,
                                                                                   margin))
                        self.metrics['validation']['PR_AUC'].append(average_precision_score(dataTargetVa,
                                                                                            margin))
                    if dataVa is not None:
                        if len(self.time) > 1:
                            last_metric = self.metrics['validation'][self.termination_metric][-1]
                            second_last_metric = self.metrics['validation'][self.termination_metric][-2]
                            if np.abs(last_metric - second_last_metric) < self.eps:
                                self.termination_reason = 'lack of sufficient decrease in validation ' + self.termination_metric
                                break
                    else:
                        if len(self.time) > 1:
                            last_metric = self.metrics['training'][self.termination_metric][-1]
                            second_last_metric = self.metrics['training'][self.termination_metric][-2]
                            if np.abs(last_metric - second_last_metric) < self.eps:
                                self.termination_reason = 'lack of sufficient decrease in training ' + self.termination_metric
                                break
            if self.termination_reason == None:
                self.termination_reason = 'model took ' + str(self.Tmax) + ' SGD steps'
            if progress:
                print('training has terminated because: ' + str(self.termination_reason))
            self.C, self.d, self.W, self.W0 = C.eval(), d.eval(), W.eval(), W0.eval()
            self.C = pd.DataFrame(self.C, index=self.cadreFts)
            self.d = pd.Series(self.d, index=self.cadreFts)
            self.W = pd.DataFrame(self.W, index=self.predictFts)
            
            ## clean up output for easier analysis
            self.metrics['training'] = pd.DataFrame(self.metrics['training'])
            if dataVa is not None:
                self.metrics['validation'] = pd.DataFrame(self.metrics['validation'])
            self.proportions = pd.concat(self.proportions, axis=1).T
            
        return self
    
    def predictFull(self, Dnew):
        """Returns classification scores/margins, predicted labels, cadre membership scores, predicted cadres, and loss"""
        if not self.fitted: print('warning: model not yet fit')
        
        tf.reset_default_graph()
        C  = tf.Variable(self.C.values, dtype=tf.float32, name='C')
        d  = tf.Variable(self.d.values, dtype=tf.float32, name='d')
        W  = tf.Variable(self.W.values, dtype=tf.float32, name='W')
        W0 = tf.Variable(self.W0, dtype=tf.float32, name='w0')
        Xcadre = tf.placeholder(dtype=tf.float32, shape=(None,self.cadreFts.shape[0]), name='Xcadre')
        Xpredict = tf.placeholder(dtype=tf.float32, shape=(None,self.predictFts.shape[0]), name='Xpredict')
        Y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='Y')
        
        ## T[n,m] = ||x^n - c^m||^2_D
        T = tf.einsum('npm,p->nm', 
              tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, Xcadre)), 
              d)
                
        ## G[n,m] = g_m(x^n)
        ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
        ## cadre-assignment scores
        G = 1 / tf.map_fn(lambda t: 
                      tf.reduce_sum(tf.exp(self.gamma*(tf.expand_dims(t,1) - 
                                             tf.expand_dims(t,0))), axis=1), T, name='G')                 

        ## E[n,y,m] = e^m_y(x^n)
        ## cadre-wise prediction scores
        E = tf.add(tf.matmul(Xpredict, W), W0, name='E')
        
        ## F[n] = f_k(x^n)
        F = tf.reduce_sum(G * E, name='F', axis=1, keepdims=True)
        
        ## predicted label
        Yhat = 0.5 * (tf.sign(F) + 1)
        ## predicted cadre
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        ## L = 1 / N sum_n log(p(y[n] | x[n])) + reg(Theta)
        error_terms = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=F)
        loss_score = tf.reduce_mean(error_terms)
        l2_d = self.lambda_d * (1 - self.alpha_d) * tf.reduce_sum(d**2)
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(W**2)
        l1_d = self.lambda_d * self.alpha_d * tf.reduce_sum(tf.abs(d))
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(tf.abs(W))
        l2_C = 1e-7 * tf.reduce_sum(C**2)
        
        loss_full = loss_score + l2_d + l2_W + l2_C + l1_d + l1_W
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            Fnew, Lnew, Gnew, mNew, loss = sess.run([F, Yhat, G, bstCd, loss_full], 
                                                     feed_dict={Xcadre: Dnew.loc[:,self.cadreFts].values,
                                                                Xpredict: Dnew.loc[:,self.predictFts].values,
                                                                Y: Dnew.loc[:,[self.targetCol]]})
        return Fnew, Lnew, Gnew, mNew, loss
   
    def predictMargin(self, Dnew):
        """Returns classification scores for new data"""
        Fnew, __, __, __, __ = self.predictFull(Dnew)
        return Fnew

    def predictClass(self, Dnew):
        """Returns predicted labels for new data"""
        __, Lnew, __, __, __ = self.predictFull(Dnew)
        return Lnew
    
    def predictCadre(self, Dnew):
        """Returns predicted cadre for new data"""
        __, __, __, mNew, __ = predictFull(Dnew)
        return mNew
    
    def entropy(self, Xnew):
        """Returns estimated entropy for each cadre"""
        __, __, G, m, __ = self.predictFull(Xnew)   
        marg = u.calcMargiProb(m, self.M)
        jont = u.calcJointProb(G, m,  self.M)
        cond = u.calcCondiProb(jont, marg)
        return u.estEntropy(cond)
    
    def weight_comparison(self):
        """Returns estimate of between-cadre prediction weight diversity: 1 / P \sum_p StdDev(w^1_p, ..., w^M_p)"""
        return self.W.std(axis=1).mean()
    
    def score(self, Dnew):
        """Returns classification rate for new data"""
        target = Dnew.loc[:,[self.targetCol]].values
        Lnew = self.predictClass(Dnew)
        return np.mean(target == Lnew)
    
    def scoreMetrics(self, Dnew):
        """Returns goodness-of-fit metrics for new data as pd.DataFrame"""
        target = Dnew.loc[:,[self.targetCol]].values        
        margin, label, __, __, loss = self.predictFull(Dnew)
        
        accuracy = np.mean(target == label)
        ROC_AUC = roc_auc_score(target, margin)
        PR_AUC = average_precision_score(target, margin)
        
        return pd.DataFrame({'loss': loss,
                             'accuracy': accuracy,
                             'ROC_AUC': ROC_AUC,
                             'PR_AUC': PR_AUC}, index=[0])
