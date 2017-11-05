## supCadresRegression.py
## author: Alexander New
## TO-DO: Modify learnCadreModel() to return the computational graph
##        Modify applyToObs() to accept a computational graph to minimize run-time
##        Force a cadre to be centered around a particular point (cadres like me)

import numpy as np
import tensorflow as tf

def eNet(alpha, lam, v):
    """Elastic-net regularization penalty"""
    return lam * (alpha * tf.reduce_sum(tf.abs(v)) + 
                  (1-alpha) * tf.reduce_sum(tf.square(v)))

def learnCadreModel(Xtr, Ytr, Xva, Yva, M, alpha, lam, inits=dict(), seed=16162):
    """Use stochastic gradient descent to learn a cadre regression model.
    Arguments: Xtr: matrix of training observations
               Ytr: matrix (one column) of training target values
               Xva: matrix of validation observations
               Yva: matrix (one column) of validation target values
               M:   number of cadres
               alpha: list of elastic net mixing hyperparameters for d, W
                      (alpha = 0 is LASSO, alpha = 1 is ridge)
               lam: list of regularization strength hyperparameters for d, W
               inits: dict of initial parameter guesses
               seed: prng seed for numpy
    Returns: dict with entries
               'fTr': training predictions
               'fVa': validation predictions
               'mTr': cadre assignments for training data
               'mVa': cadre assignments for validation data
               'loss': loss function values for training, validation data
               'C', 'd', 'W', 'w0', 's': optimal model parameters
               'Gtr', 'Gva': matrices of cadre membership weights
    """
    np.random.seed(seed)
######################
## model parameters ##
######################
    gamma = 10          # cadre-assignment sharpness parameter
    Tmax = 10000        # number of iterations
    recd = Tmax // 100  # interval for record-keeping in sgd 
    eta  = 1e-3         # SGD step length
    Nba  = 50           # minibatch size for SGD
    eps  = 1e-3         # tolerance criterion
    
    Ntr, P = Xtr.shape  # number of (training observations, features)

    errTr, errVa = [], [] # used to store training and validation errors
############################################
## tensorflow parameters and placeholders ##
############################################
    tf.reset_default_graph()

    ## cadre centers parameter
    if 'C' in inits:
        C = tf.Variable(inits['C'], dtype=tf.float64, name='C')
    else:
        C = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(P,M)), 
                        dtype=tf.float64, name='C')
    ## cadre determination weights parameter
    if 'd' in inits:
        d = tf.Variable(inits['d'], dtype=tf.float64, name='d')
    else:
        d = tf.Variable(np.random.uniform(size=(P)), dtype=tf.float64, name='d')
    ## regression hyperplane weights parameter
    if 'W' in inits:
        W = tf.Variable(inits['W'], dtype=tf.float64, name='W')
    else:
        W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(P,M)), 
                        dtype=tf.float64, name='W')
    ## regression hyperplane bias parameter
    if 'w0' in inits:
        w0 = tf.Variable(inits['w0'], dtype=tf.float64, name='w0')
    else:
        w0 = tf.Variable(tf.zeros(shape=(M,), dtype=tf.float64), 
                         dtype=tf.float64, name='w0')
    ## model error parameter
    if 'sigma' in inits:
        sigma = tf.Variable(inits['sigma'], dtype=tf.float64, name='sigma')
    else:
        sigma = tf.Variable(0.1, dtype=tf.float64, name='sigma')

    X = tf.placeholder(dtype=tf.float64, shape=(None,P), name='X')
    N = tf.cast(tf.gather(tf.shape(X), 0), dtype=tf.float64, name='N')
    Y = tf.placeholder(dtype=tf.float64, shape=(None), name='Y')

    ## T[n,m] = ||x^n - c^m||^2_D
    T = tf.einsum('npm,p->nm', 
              tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, X)), 
              tf.abs(d))

    ## G[n,m] = g_m(x^n)
    ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
    G = 1 / tf.map_fn(lambda t: 
                  tf.reduce_sum(tf.exp(gamma*(tf.expand_dims(t,1) - 
                                         tf.expand_dims(t,0))), axis=1), T, name='G')                 

    ## E[n,m] = e_m(x^n)
    E = tf.add(tf.matmul(X, W), w0, name='E')
    
    ## f[n] = f(x^n)
    F = tf.reduce_sum(G * E, axis=1, name='F') # this won't work if minibatch size 1 is used

    ## L = 1 / N sum_n sum_m g_m(x^n) * (e_m(x^n) - y_n) ^2
    L = tf.add(tf.reduce_mean(tf.reduce_sum(G * (E - Y) **2, axis=1)) / 2 / sigma**2,
         (eNet(alpha[0], lam[0], d) + eNet(alpha[1], lam[1], W)) / 2 / N / sigma**2 +
         (1/2 + 1/N) * tf.log(sigma**2), name='L')

    bstCd = tf.argmax(G, axis=1, name='bestCadre')
    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(L)

####################
## learning model ##
####################
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
    
        ## perform optimization
        for t in range(Tmax):
            inds = np.random.choice(Ntr, Nba, replace=False)      
            sess.run(optimizer,  feed_dict={X: Xtr[inds,:],
                                            Y: Ytr[inds]})
            # record-keeping        
            if not t % recd:
                errTr.append(L.eval(feed_dict={
                        X: Xtr,
                        Y: Ytr}))
                errVa.append(L.eval(feed_dict={
                        X: Xva,
                        Y: Yva}))
                if len(errTr) > 2 and np.abs(errTr[-1] - errTr[-2]) < eps:
                    break
    
        ## calculate target predictions
        FeTr = F.eval(feed_dict={X: Xtr,
                                 Y: Ytr})
        FeVa = F.eval(feed_dict={X: Xva,
                                 Y: Yva})
        ## calculate cadre identity predictions
        mTr = bstCd.eval(feed_dict={X: Xtr,
                                    Y: Ytr})
        mVa = bstCd.eval(feed_dict={X: Xva,
                                    Y: Yva})
        ## calculate cadre membership weights
        GeTr = G.eval(feed_dict={X: Xtr, Y: Ytr})
        GeVa = G.eval(feed_dict={X: Xva, Y: Yva})

        ## evaluate optimal parameters
        Ce, de, We, w0e, Se = C.eval(), d.eval(), W.eval(), w0.eval(), sigma.eval()

        modelOutput = {'fTr': FeTr, 'fVa': FeVa, 'mTr': mTr, 'mVa': mVa, 'loss': (errTr[-1], errVa[-1]),
                       'C': Ce, 'd': de, 'W': We, 'w0': w0e, 's': Se, 'Gtr': GeTr, 'Gva': GeVa}
    return modelOutput

def applyToObs(params, Xnew):
    """Apply a cadre model to a new set of observations
    Arguments: params: dict with entries 'C', 'd', 'W', 'w0'
               Xnew: matrix of new observations
    Returns: dict with entries 'F': predicted values
                               'G': cadre membership weights
                               'm': cadre assignments
    """
    gamma = 10        # cadre assignment sharpness parameter
    P = Xnew.shape[1] # number of features
    ## load model information and set up input placeholder
    tf.reset_default_graph()
    C  = tf.Variable(params['C'], dtype=tf.float64, name='C')
    d  = tf.Variable(params['d'], dtype=tf.float64, name='d')
    W  = tf.Variable(params['W'], dtype=tf.float64, name='W')
    w0 = tf.Variable(params['w0'], dtype=tf.float64, name='w0')
    X = tf.placeholder(dtype=tf.float64, shape=(None,P), name='X')
    
    ## T[n,m] = ||x^n - c^m||^2_D
    T = tf.einsum('npm,p->nm', 
              tf.square(tf.map_fn(lambda x: tf.expand_dims(x,1) - C, X)), 
              tf.abs(d))

    ## G[n,m] = g_m(x^n)
    ##        = 1 / sum_m' exp(gamma(T[n,m] - T[n,m']))
    G = 1 / tf.map_fn(lambda t: 
                  tf.reduce_sum(tf.exp(gamma*(tf.expand_dims(t,1) - 
                                tf.expand_dims(t,0))), axis=1), T, name='G')                 

    ## E[n,m] = e_m(x^n)
    E = tf.add(tf.matmul(X, W), w0, name='E')
    
    ## f[n] = f(x^n)
    F = tf.reduce_sum(G * E, axis=1, name='F') # this won't work if minibatch size 1 is used
    bstCd = tf.argmax(G, axis=1, name='bestCadre')
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        Fnew, Gnew, mNew = sess.run([F, G, bstCd], feed_dict={X: Xnew})
    predictionOutput = {'F': Fnew, 'G': Gnew, 'm': mNew}
    return predictionOutput
