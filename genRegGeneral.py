## genClsBin.py

## tensorboard? localhost:6006 to get there

import numpy as np
import tensorflow as tf

#np.random.seed(65)

##########################
## function definitions ##
##########################

## elastic net penalty
def eNet(alpha, lam, v):
    return lam * (alpha * tf.reduce_sum(tf.abs(v)) + 
                  (1-alpha) * tf.reduce_sum(tf.square(v)))

def learnClassifier(Xtr, Ytr, Xva, Yva, M, alpha, lam, seed):
    np.random.seed(seed)
######################
## model parameters ##
######################
    gamma = 10         # cadre-assignment sharpness parameter

    Tmax = 10000 # number of iterations
    recd = Tmax // 100  # interval for record-keeping in sgd 
    eta  = 1e-3 # step length
    Nba  = 50   # minibatch size for SGD
#    Nba = 900
    eps  = 1e-3 # tolerance criterion
#    eps = 1e-4
    
    Ntr, P = Xtr.shape

    errTr, errVa = [], []
############################################
## tensorflow parameters and placeholders ##
############################################
    tf.reset_default_graph()

    C  = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(P,M)), 
                 dtype=tf.float64, name='C')
    d  = tf.Variable(np.random.uniform(size=(P)), dtype=tf.float64, name='d')
    Theta  = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(P,M)), 
                     dtype=tf.float64, name='Theta')
    Theta0 = tf.Variable(tf.zeros(shape=(M,), dtype=tf.float64), dtype=tf.float64,
                     name='Theta0')
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
    E = tf.add(tf.matmul(X, Theta), Theta0, name='E')
    
    ## f[n] = f(x^n)
    F = tf.reduce_sum(G * E, axis=1, name='F') # this won't work if minibatch size 1 is used

    ## L = 1 / N sum_n sum_m g_m(x^n) * (e_m(x^n) - y_n) ^2
    L = tf.add(tf.reduce_mean(tf.reduce_sum(G * (E - Y) **2, axis=1)) / 2 / sigma**2,
         (eNet(alpha[0], lam[0], d) + eNet(alpha[1], lam[1], Theta)) / 2 / N / sigma**2 +
         (1/2 - 1/N) * tf.log(sigma**2), name='L')

    bstCd = tf.argmax(G, axis=1, name='bestCadre')
    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(L)

####################
## learning model ##
####################
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
#        tf.local_variables_initialize
        
        sess.run(tf.local_variables_initializer())
    
        ## perform optimization
        for t in range(Tmax):
            inds = np.random.choice(Ntr, Nba, replace=False)      
            sess.run(optimizer,  feed_dict={X: Xtr[inds,:],
                                       Y   : Ytr[inds]})
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
    
        ## calculate final predictions
        FeTr = F.eval(feed_dict={X: Xtr,
                                 Y: Ytr})
        FeVa = F.eval(feed_dict={X: Xva,
                                 Y: Yva})
        mTr = bstCd.eval(feed_dict={X: Xtr,
                                    Y: Ytr})
        mVa = bstCd.eval(feed_dict={X: Xva,
                                    Y: Yva})
        GeTr = G.eval(feed_dict={X: Xtr, Y: Ytr})
        GeVa = G.eval(feed_dict={X: Xva, Y: Yva})
        ## get descriptive statistics about each cadre
#        cadMeansTr, cadMeansVa = np.zeros((M,P)), np.zeros((M,P))
#        cadVarisTr, cadVarisVa = np.zeros((M,P)), np.zeros((M,P))
#        cadCounts = np.zeros(shape=(M,2))
#        for m in range(M):
#            cadCounts[m,0], cadCounts[m,1] = np.sum(mTr==m), np.sum(mVa==m)
#            if sum(mTr==m):
#                cadMeansTr[m,:] = np.mean(Xtr[mTr==m,:], axis=0)
#                cadVarisTr[m,:] = np.var(Xtr[mTr==m,:], axis=0)
#            if sum(mVa==m):
#                cadMeansVa[m,:] = np.mean(Xva[mVa==m,:], axis=0)
#                cadVarisVa[m,:] = np.var(Xva[mVa==m,:], axis=0)
#        GeVa = G.eval(feed_dict={Xcad: Xva[:,cadFts],
#                             Xtar: Xva[:,tarFts],
#                             Y   : Yva})

        Ce, de, Te, Te0, Se = C.eval(), d.eval(), Theta.eval(), Theta0.eval(), sigma.eval()

        things = {'fTr': FeTr, 'fVa': FeVa, 'mTr': mTr, 'mVa': mVa, 'loss': (errTr[-1], errVa[-1]),
                  'C': Ce, 'd': de, 'T': Te, 'T0': Te0, 's': Se, 'Gtr': GeTr, 'Gva': GeVa}
    return things
