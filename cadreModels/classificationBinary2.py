## classificationBinary.py
## binary classification cadres with logistic loss
## parameters have both l1 and l2 regularization, and l1 terms are handled with
## proximal gradient steps
## input is a pandas.DataFrame containing both target and input features

from __future__ import division, print_function, absolute_import

import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score, average_precision_score

def eNet(alpha, lam, v):
    """Elastic-net regularization penalty"""
    return lam * (alpha * tf.reduce_sum(tf.abs(v)) + 
                  (1-alpha) * tf.reduce_sum(tf.square(v)))
    
class binaryCadreModel(object):
    
    def __init__(self, M=2, gamma=10., lambda_d=0.01, lambda_W=0.01,
                 alpha_d=0.9, alpha_W=0.9, Tmax=10000, record=100, 
                 eta=2e-3, Nba=50, eps=1e-3, loss_type='combined'):
        ## hyperparameters / structure
        self.M = M                # number of cadres
        self.gamma = gamma        # cadre assignment sharpness
        self.lambda_d = lambda_d  # regularization strengths
        self.lambda_W = lambda_W
        self.alpha_d = alpha_d    # elastic net mixing weights
        self.alpha_W = alpha_W    
        self.fitted = False
        self.loss_type = loss_type
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
        self.norms = {'W': [],
                      'd': [],
                      'C': []}
        self.time = [] # times
        self.proportions = [] # cadre membership proportions during training
    
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
            seed=16162, store=False, progress=False, decrease_stepsize=True,
           get_norms=False):
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
        ## map from y \in {0, 1} to y \in {-1, +1} if separate loss is used
        if self.loss_type == 'separate':
            dataTarget = 2 * dataTarget - 1
        
        if dataVa is not None:
            dataCadreVa = dataVa.loc[:,self.cadreFts].values
            dataPredictVa = dataVa.loc[:,self.predictFts].values
            dataTargetVa = dataVa.loc[:,[self.targetCol]].values
            if self.loss_type == 'separate':
                dataTargetVa = 2 * dataTargetVa - 1
        
        
        ############################################
        ## tensorflow parameters and placeholders ##
        ############################################
        tf.reset_default_graph()
    
        ## cadre centers parameter
        C = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Pcadre,self.M)), 
                            dtype=tf.float32, name='C')
        ## cadre determination weights parameter
        d = tf.Variable(np.random.uniform(size=(Pcadre)), dtype=tf.float32, name='d')
        
        ## regression hyperplane weights parameter
        W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Ppredict,self.M)), 
                            dtype=tf.float32, name='W')
        ## regression hyperplane bias parameter
        W0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='W0')
    
        Xcadre = tf.placeholder(dtype=tf.float32, shape=(None,Pcadre), name='Xcadre')
        Xpredict = tf.placeholder(dtype=tf.float32, shape=(None,Ppredict), name='Xpredict')
        Y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='Y')
        eta = tf.placeholder(dtype=tf.float32, shape=(), name='eta')
        
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
        bstCd = tf.argmax(G, axis=1, name='bestCadre')

        ## E[n,y] = e^m_y(x^n)
        ## cadre-wise prediction scores
        E = tf.add(tf.matmul(Xpredict, W), W0, name='E')
        
        if self.loss_type == 'combined':
            ## F[n] = f_k(x^n)
            F = tf.reduce_sum(G * E, name='F', axis=1, keepdims=True)

            ## L = 1 / N sum_n log(p(y[n] | x[n])) + reg(Theta)
            error_terms = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=F)
            
            loss_score = tf.reduce_mean(error_terms)
        
        elif self.loss_type == 'separate':
            error_terms = tf.transpose(tf.nn.sigmoid_cross_entropy_with_logits(
                                       labels=tf.squeeze(Y), logits=tf.transpose(E)))
            
            F = tf.reduce_sum(G * tf.nn.sigmoid(E), axis=1, keepdims=True)
            
            loss_score = tf.reduce_mean(tf.reduce_sum(G * error_terms, axis=1))
        
        l2_d = self.lambda_d * (1 - self.alpha_d) * tf.reduce_sum(d**2)
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(W**2)
        l1_d = self.lambda_d * self.alpha_d * tf.reduce_sum(tf.abs(d))
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(tf.abs(W))
        l2_C = 1e-7 * tf.reduce_sum(C**2)
        
        ## loss that is fed into optimizer
        loss_opt = loss_score + l2_d + l2_W + l1_d + l1_W + l2_C
        ## full loss, including l1 terms handled with proximal gradient
        loss_full = loss_opt + l1_d + l1_W
        optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_opt)
        
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
                ## take SGD step
                if decrease_stepsize:
                    eta_t = self.eta / np.sqrt(t+1)
                else:
                    eta_t = self.eta
                sess.run(optimizer, feed_dict={Xcadre: dataCadre[inds,:],
                                               Xpredict: dataPredict[inds,:],
                                               Y: dataTarget[inds,:],
                                               eta: eta_t})
                ## take proximal gradient step
                sess.run([thresh_d, thresh_W], feed_dict={eta: eta_t})
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
                    l, margin, cadres = sess.run([loss_full, F, bstCd], 
                                                 feed_dict={Xcadre: dataCadre,
                                                            Xpredict: dataPredict,
                                                            Y: dataTarget})
                    if self.loss_type == 'combined':
                        yhat = 0.5 * (np.sign(margin) + 1)
                    elif self.loss_type == 'separate':
                        yhat = 2 * np.round(margin) - 1
                    self.metrics['training']['loss'].append(l)
                    self.metrics['training']['accuracy'].append(np.mean(yhat == dataTarget))
                    self.metrics['training']['ROC_AUC'].append(roc_auc_score(dataTarget,
                                                                             margin))
                    self.metrics['training']['PR_AUC'].append(average_precision_score(dataTarget,
                                                                                      margin))
                    self.proportions.append(pd.Series(cadres).value_counts().T)
                    self.proportions[-1] /= self.proportions[-1].sum()
                    
                    if get_norms:
                        C_t, d_t, W_t = sess.run([C, d, W])
                        self.norms['C'].append(pd.DataFrame(np.linalg.norm(C_t, axis=0)[None,:], 
                                                            columns=['c'+str(m) for m in range(self.M)]))
                        self.norms['W'].append(pd.DataFrame(np.linalg.norm(W_t, axis=0)[None,:],
                                                            columns=['w'+str(m) for m in range(self.M)]))
                        self.norms['d'].append(pd.Series(np.linalg.norm(d_t)))
                        
                    if dataVa is not None:
                        l, margin = sess.run([loss_full, F], feed_dict={Xcadre: dataCadreVa,
                                                                Xpredict: dataPredictVa,
                                                                Y: dataTargetVa})
                        if self.loss_type == 'combined':
                            yhat = 0.5 * (np.sign(margin) + 1)
                        elif self.loss_type == 'separate':
                            yhat = 2 * np.round(margin) - 1
                        self.metrics['validation']['loss'].append(l)
                        self.metrics['validation']['accuracy'].append(np.mean(yhat == dataTargetVa))
                        self.metrics['validation']['ROC_AUC'].append(roc_auc_score(dataTargetVa,
                                                                                   margin))
                        self.metrics['validation']['PR_AUC'].append(average_precision_score(dataTargetVa,
                                                                                            margin))
                        
            self.C, self.d, self.W, self.W0 = C.eval(), d.eval(), W.eval(), W0.eval()
            self.C = pd.DataFrame(self.C, index=self.cadreFts)
            self.d = pd.Series(self.d, index=self.cadreFts)
            self.W = pd.DataFrame(self.W, index=self.predictFts)
            
            ## clean up output for easier analysis
            self.metrics['training'] = pd.DataFrame(self.metrics['training'])
            if dataVa is not None:
                self.metrics['validation'] = pd.DataFrame(self.metrics['validation'])
            self.proportions = pd.concat(self.proportions, axis=1).T
            if get_norms:
                self.norms = {key: pd.concat(value) for key, value in self.norms.items()}
            
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
        
        if self.loss_type == 'combined':
            target = Dnew.loc[:,[self.targetCol]].values
        elif self.loss_type == 'separate':
            target = 2 * Dnew.loc[:,[self.targetCol]].values - 1
        
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
        bstCd = tf.argmax(G, axis=1, name='bestCadre')
        
        if self.loss_type == 'combined':
            F = tf.reduce_sum(G*E, name='F', axis=1, keepdims=True)
            ## values of Yhat are in {0, 1}
            Yhat = 0.5 * (tf.sign(F) + 1)
            
            ## L = 1 / N sum_n log(p(y[n] | x[n])) + reg(Theta)
            error_terms = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=F)
            
            loss_score = tf.reduce_mean(error_terms)
        elif self.loss_type == 'separate':
            error_terms = tf.transpose(tf.nn.sigmoid_cross_entropy_with_logits(
                                       labels=tf.squeeze(Y), logits=tf.transpose(E)))
            
            F = tf.reduce_sum(G * tf.nn.sigmoid(E), axis=1, keepdims=True)
            ## values of Yhat are in {-1, +1}
            Yhat = 2 * tf.round(F) - 1
            
            loss_score = tf.reduce_mean(tf.reduce_sum(G * error_terms, axis=1))
        
        l2_d = self.lambda_d * (1 - self.alpha_d) * tf.reduce_sum(d**2)
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(W**2)
        l1_d = self.lambda_d * self.alpha_d * tf.reduce_sum(tf.abs(d))
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(tf.abs(W))
        l2_C = 1e-7 * tf.reduce_sum(C**2)
        
        loss_full = loss_score + l2_d + l2_W + l2_C + l1_d + l1_W
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            f, e, err = sess.run([F, E, error_terms], feed_dict={Xcadre: Dnew.loc[:,self.cadreFts].values,
                                                                Xpredict: Dnew.loc[:,self.predictFts].values,
                                                                Y: target})
#             print('f', f.min(), f.max())
#             print('e', e.min(), e.max())
#             print('err', err.min(), err.max())
            
            Fnew, Lnew, Gnew, mNew, loss = sess.run([F, Yhat, G, bstCd, loss_full], 
                                                     feed_dict={Xcadre: Dnew.loc[:,self.cadreFts].values,
                                                                Xpredict: Dnew.loc[:,self.predictFts].values,
                                                                Y: target})
        return Fnew, Lnew, Gnew, mNew, loss
   
    def predictClass(self, Dnew):
        __, Lnew, __, __ = self.predictFull(Dnew)
        return Lnew
    
    def score(self, Dnew):
        target = Dnew.loc[:,[self.targetCol]].values
        __, Lnew, __, __ = self.predictFull(Dnew)
        return np.mean(target == Lnew)
    
    def scoreMetrics(self, Dnew):
        ## values of target are in {0, 1}
        if self.loss_type == 'combined':
            target = Dnew.loc[:,[self.targetCol]].values
        ## values of target are in {-1, +1}
        elif self.loss_type == 'separate':
            target = 2 * Dnew.loc[:,[self.targetCol]].values - 1        
        margin, label, __, __, loss = self.predictFull(Dnew)
#         print('target', target.shape, np.unique(target))
#         print('label', label.shape, np.unique(label))
#         print('margin', margin.shape, margin.min(), margin.max())
        
        accuracy = np.mean(target == label)
        ROC_AUC = roc_auc_score(target, margin)
        PR_AUC = average_precision_score(target, margin)
        
        return pd.DataFrame({'loss': loss,
                             'accuracy': accuracy,
                             'ROC_AUC': ROC_AUC,
                             'PR_AUC': PR_AUC}, index=[0])
