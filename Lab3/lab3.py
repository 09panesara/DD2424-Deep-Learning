from utils import CIFAR10Data, plot_costs_accuracies
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


''' K-Layer Classifier'''
class Layer:
    def __init__(self, input_dim, output_dim, activation='relu', batch_normalisation=False, sig=0.1, dropout_thresh=None, initialisation='He'):
        if batch_normalisation:
            # initialise weights to be normally distributed with sigmas equal to sig at each layer
            if initialisation == 'He':
                print('He')
                self.W = np.random.normal(loc=0, scale=1/np.sqrt(input_dim), size=(output_dim, input_dim)) 
            else:
#                 print('initialising with sigma', sig)
                self.W = np.random.normal(loc=0, scale=sig, size=(output_dim, input_dim)) 
            self.gamma  = np.ones((output_dim, 1))
            self.beta   =  np.zeros((output_dim, 1))
            self.means_avg = None
            self.sigmas_avg = None
        else:
            # use He initialization 
            self.W = np.random.normal(loc=0, scale=1/np.sqrt(input_dim), size=(output_dim, input_dim)) 
        self.b = np.zeros((output_dim,1))
        self.activation_fn = activation # should be softmax for last layer in definition
        self.apply_bn = batch_normalisation
        self.S_hat = None # batch normalised scores (pre scale and shift)
        self.dropout_thresh = dropout_thresh # percentage of nodes to drop
        if self.dropout_thresh is not None:
            print('Applying dropout p=', 1-self.dropout_thresh)
        

        
    def activation(self, S):
        if self.activation_fn == 'relu':
            return self.relu(S) 
        elif self.activation_fn == 'softmax':
            return self.softmax(S)
        else:
            raise Exception('activation function invalid') 
    
    
    def relu(self, X):
        return np.where(X >= 0, X, 0)
    
    
    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=0)
    
    
    def forward_pass(self, X, last_layer=False, is_train=False, alpha=0.9):
        if self.dropout_thresh is not None:
            D, N = X.shape
            indices = np.ones(D)
            number_to_zero = D*self.dropout_thresh
            indices[-int(number_to_zero):] = 0
            u1_batch = np.asarray([np.random.permutation(indices) for i in range(N)]).T
            ''' divide by p = 1-dropout_thresh = proportion of connections to keep 
                so that don't have to compensate at test'''
            u1_batch /= (1-self.dropout_thresh) 
            X = np.multiply(X, u1_batch)
            
        S = np.dot(self.W, X) 
        S += self.b
        self.S = S

        if self.apply_bn and not last_layer:
            
            if is_train:
                self.means = np.sum(S, axis=1).reshape(-1,1) / S.shape[1] 
                self.sigmas =  np.sum((S.T-self.means.T)**2, axis=0).reshape(-1,1) / S.shape[1]
                if self.means_avg is None:
                    self.means_avg = np.copy(self.means)
                    self.sigmas_avg = np.copy(self.sigmas)
                else:
                    self.means_avg = alpha * self.means_avg + (1-alpha) * self.means
                    self.sigmas_avg = alpha * self.sigmas_avg + (1-alpha) * self.sigmas
                
                assert self.means.shape == self.sigmas.shape
                means = self.means
                sigmas = self.sigmas
            else: # at test time, use averages
                means = self.means_avg 
                sigmas = self.sigmas_avg
            
            # batch normalise

            self.S_hat = (self.S - means)/(np.sqrt(sigmas+1e-10))

            S_scaled = np.multiply(self.gamma, self.S_hat) + self.beta # to do sort these out
            self.S_scaled = S_scaled
            return self.activation(S_scaled)
        else:
            return self.activation(S)
        
        
    def batch_norm_back_pass(self, G, eps=1e-10):   
        N = G.shape[1] 
        sigma1 = np.power(self.sigmas+eps, -0.5)
        sigma2 = np.power(self.sigmas+eps, -1.5)
        G1 = np.multiply(G, np.dot(sigma1,np.ones((1,N))))
        G2 = np.multiply(G, np.dot(sigma2, np.ones((1,N))))
        
        D = self.S - self.means
        c = np.dot(np.multiply(G2, D), np.ones((N,1)))
        G = G1 - np.dot(np.dot(G1,np.ones((N,1))), np.ones((1,N)))/N - np.multiply(D, np.dot(c, np.ones((1,N))))/N
                                                                       
        return G
        
    
    def update_gradients(self, eta, grad_W, grad_b, grad_gamma=None, grad_beta=None):
        self.W -= (eta * grad_W)
        self.b -= (eta * grad_b)
        if grad_gamma is not None:
            self.gamma -= (eta * grad_gamma)
        if grad_beta is not None:
            self.beta -= (eta * grad_beta)
  

class KLayerClassifier:
    def __init__(self, batch_size=100, eta=0.001, n_cycles=2, lamda=0, cyclical_lr=False, n_s=None, eta_min=None, eta_max=None, lr_range_test=False, dropout_thresh=None, batch_normalisation=False):
        self.batch_size = batch_size # number of images in a batch
        self.batch_normalisation = batch_normalisation
        self.lamda = lamda
        
        self.cyclical_lr = cyclical_lr
        if cyclical_lr:
            self.batch_size = 100
            self.l = 0
            self.n_s = n_s # step size
            self.eta_min = eta_min
            self.eta_max = eta_max
            self.plot_every_n_steps = int(2*n_s/10)
            self.n_cycles = n_cycles
            
        else:
            self.eta = eta
        
        if self.batch_normalisation:
            print('Performing batch normalisation', batch_normalisation)
            self.backward_pass = self.backward_pass_w_bn
            
            self.alpha = 0.9 # exponential moving average 
            
        self.lr_range_test = lr_range_test # whether to perform learning rate range test to see what eta_min and eta_max should be
        self.plot_every_n_steps = batch_size 
        self.dropout_thresh = dropout_thresh # proportion of nodes to switch off ( =(1-p) from slides)
        
        # initialise layers
        self.layers = None
    
    def __reshape(self, X):
        return X.reshape((X.shape[0],1))
        
        
    def build_model(self, input_dim, hidden_nodes, dropout_threshes=None, initialisation='He', sigma=0.01):
        if dropout_threshes is None:
            dropout_threshes = [None]*len(hidden_nodes)

        self.add_layer(input_dim=input_dim, output_dim=hidden_nodes[0], dropout_thresh=dropout_threshes[0], \
                       initialisation=initialisation, sigma=sigma)

        for i, h in enumerate(hidden_nodes[:-1]):
            self.add_layer(input_dim=hidden_nodes[i], output_dim=hidden_nodes[i+1], \
                           dropout_thresh=dropout_threshes[i], initialisation=initialisation, sigma=sigma)
    
        # last layer with output dim number of classes
        self.add_layer(input_dim=hidden_nodes[-1], output_dim=10, activation='softmax', initialisation=initialisation, sigma=sigma)
        
        
    def add_layer(self, input_dim, output_dim, activation='relu', dropout_thresh=None, initialisation='He', sigma=0.01):
        # add a new layer
        if self.layers is None:
            self.layers = []
        self.layers.append(Layer(input_dim, output_dim, activation, batch_normalisation=self.batch_normalisation, \
                                 dropout_thresh=dropout_thresh, initialisation=initialisation, sig=sigma))
        

    def normalise(self, train_X, val_X, test_X):
        ''' X has shape (d,n) where d = dimensionality of each image, n is number of images '''
        mean = np.mean(train_X, axis=1)
        std = np.std(train_X, axis=1)
        original_shape = train_X.shape
        # apply same transformation to all of the datasets using params from train set
        def _normalise_helper(a, m, s):
            return ((a.T - m.T) / s.T).T
        
        train_X = _normalise_helper(train_X, mean, std)
        val_X = _normalise_helper(val_X, mean, std)
        test_X = _normalise_helper(test_X, mean, std)
        return train_X, val_X, test_X

    
    def forward_pass(self, X, is_train=True):
        H = np.copy(X)
        Xs = [X]

        for layer in self.layers[:-1]: # loop layers 1...k-1
            H = layer.forward_pass(H, is_train=is_train)
            Xs.append(H)
        # apply softmax to last layer instead of relu
        last_layer = self.layers[-1]
        P = last_layer.forward_pass(H, last_layer=True, is_train=is_train)
        assert len(Xs) == len(self.layers) # k
        return Xs, P
    
    
    def compute_accuracy(self, X, Y):
        ''' X is data (dim, N), y is gt (C, N), W is weight matrix, b is bias, Y is 1hot encoded labels'''
        _, P = self.forward_pass(X, is_train=False)
        pred = np.argmax(P, axis=0)
        lbls = np.argmax(Y, axis=0)
        accuracy = np.mean(pred == lbls)
        return pred, accuracy
    

    def compute_cost(self, X, Y): # TODO
        ''' 
            X: dxn (dimensionality by # images)
            Y: Kxn (no. classes one-hot encoded by # images)
            J: scalar corresponding to sum of loss of ntwks predictions of X relative to gt labels 
        '''
        _, P = self.forward_pass(X)
        N = X.shape[1]
        loss = -np.sum(Y*np.log(P)) / N
        
        regularisation_term = 0
        for layer in self.layers:
            regularisation_term += np.sum(layer.W**2)
        
        cost = loss + self.lamda * regularisation_term
    
        return loss, cost
        
    
    def backward_pass(self, Xs, Y, P):
        G = -(Y - P)
        N = G.shape[1]
        k = len(self.layers)
        
        grad_Ws = []
        grad_bs = []
        
        for l in range(k-1, 0, -1): # propagate gradient backwarsd to first layer 
            layer = self.layers[l]
            H = Xs[l] # previous layers X (index at l due to appending input at start for first layer in forward pass)
            grad_W = np.dot(G, H.T) / N + 2 * self.lamda * layer.W 
            grad_b = self.__reshape(np.sum(G, axis=1) / N)

            grad_Ws.append(grad_W)
            grad_bs.append(grad_b)
            G = np.dot(layer.W.T, G)
            Ind = H > 0
            G = np.multiply(G, Ind)
            
        # For first layer
        layer = self.layers[0]
        grad_W = np.dot(G, Xs[0].T) / N + 2 * self.lamda * layer.W 
        grad_b = self.__reshape(np.sum(G, axis=1) / N)
        grad_Ws.append(grad_W)
        grad_bs.append(grad_b)
        
        # reverse order from first layer to last
        return grad_Ws[::-1], grad_bs[::-1], [None]*len(self.layers), [None]*len(self.layers)
    

    def backward_pass_w_bn(self, Xs, Y, P):
        G = -(Y - P)

        N = G.shape[1]
        k = len(self.layers)
        H = Xs[-1]
        
        # last layer doesn't have batch normalisation
        layer = self.layers[-1]
        
        grad_W = np.dot(G, H.T) / N + 2*self.lamda * layer.W
        grad_b = self.__reshape(np.sum(G, axis=1) / N)
        
        grad_Ws = [grad_W]
        grad_bs = [grad_b]
        grad_gammas = [None]
        grad_betas = [None]
        
        
        for l in range(k-2, -1, -1): # propagate gradient backwarsd to first layer from layers k-1 to 0
            layer = self.layers[l]
            
            # Propagate G to current layer
            G = np.dot(self.layers[l+1].W.T, G)
            G = np.multiply(G, H>0)
            
            H = Xs[l]
            # compute gradient for scale and offset parameters
            grad_gamma = np.dot(np.multiply(G, layer.S_hat), np.ones((N,1))) / N 
            grad_beta  = np.dot(G, np.ones((N,1))) / N
            # propagate gradients through scale and shift
            G = np.multiply(G, layer.gamma)
            # propagate G through batch normalisation
            G = layer.batch_norm_back_pass(G)  
            
            # calculate gradients of J wrt bias and weights
            grad_W = np.dot(G, H.T) / N 
            
            grad_W += (2 * self.lamda * layer.W )
            grad_b = self.__reshape(np.sum(G, axis=1) / N)
            
            grad_Ws.append(grad_W)
            grad_bs.append(grad_b)
            grad_gammas.append(grad_gamma)
            grad_betas.append(grad_beta)
            
   
        # reverse order from first layer to last
        return grad_Ws[::-1], grad_bs[::-1], grad_gammas[::-1], grad_betas[::-1] 
    
    
    def compute_gradients(self, X, Y):
        ''' computes gradients of the cost function wrt W and b for batch X '''
        N = X.shape[1]

        # forward pass, apply dropout if its set
        Xs, P = self.forward_pass(X)
        
        # backward pass
        grad_Ws, grad_bs, grad_gammas, grad_betas = self.backward_pass(Xs, Y, P)
        
        return grad_Ws, grad_bs, grad_gammas, grad_betas
    
        
        
    def train(self, X, Y, random_shuffle=False, val_X=None, val_Y=None, get_accuracies_costs=False, max_lr=0.1, n_epochs=20, apply_jitter=False):
        n = X.shape[1]
        
        number_of_batches = int(n / self.batch_size)
        
        assert number_of_batches > 0
        indices = np.arange(X.shape[1])
        if random_shuffle:
            print('Randomly shuffling')
            
        if apply_jitter:
            print('Adding jitter')
        
        if self.cyclical_lr:
            print('Cyclical learning rate')
            n_epochs = int(self.n_s * 2 * self.n_cycles / number_of_batches)
       
        accuracies = {'train': [], 'val': []}
        costs = {'train': [], 'val': []}
        losses = {'train': [], 'val': []}
        update_steps = []
        
        eta = self.eta if not self.cyclical_lr else self.eta_min
        t = 0 # each update step % 2*n_s
    
        if self.lr_range_test:
            eta = 0
            self.plot_every_n_step = int(n_epochs * number_of_batches / 20)
            eta_incr = max_lr * self.plot_every_n_step / (n_epochs * number_of_batches)
            
        print('total epochs', n_epochs)
        for epoch in range(n_epochs):  
            if epoch % 10 == 0:
                print('epoch', epoch)
            if random_shuffle:
                np.random.shuffle(indices)
                X = np.take(X, indices, axis=1)
                Y = np.take(Y, indices, axis=1)
                
            for j in range(number_of_batches):
                if self.cyclical_lr:  
                    t_mod = t % (2*self.n_s)
                    if t_mod <= self.n_s:
                        eta = self.eta_min + t_mod/self.n_s * (self.eta_max - self.eta_min)

                    elif t_mod <= 2*self.n_s:
                        eta = self.eta_max - (t_mod - self.n_s)/self.n_s * (self.eta_max - self.eta_min)
   

                j_start = j * self.batch_size
                j_end = (j+1) * self.batch_size
                Xbatch = X[:, j_start:j_end]
                Ybatch = Y[:, j_start:j_end]
                
                if apply_jitter:
                    Xbatch = add_jitter(Xbatch)
                
                # Perform MiniBatch Gradient Descent
                grad_Ws, grad_bs, grad_gammas, grad_betas = self.compute_gradients(Xbatch, Ybatch)
                
                for i in range(len(self.layers)): 
                    self.layers[i].update_gradients(eta, grad_Ws[i], grad_bs[i], grad_gammas[i], grad_betas[i]) 

                if get_accuracies_costs and (t % self.plot_every_n_steps == 0):
                    _, train_accuracy = self.compute_accuracy(X, Y)
                    _, val_accuracy = self.compute_accuracy(val_X, val_Y)
                    accuracies['train'].append(train_accuracy)
                    accuracies['val'].append(val_accuracy)
                    train_loss, train_cost = self.compute_cost(X, Y)
                    val_loss, val_cost = self.compute_cost(val_X, val_Y)
                    costs['train'].append(train_cost)
                    costs['val'].append(val_cost)
                    losses['train'].append(train_loss)
                    losses['val'].append(val_loss)
                    update_steps.append(t)
            
                
                if self.lr_range_test and (t % self.plot_every_n_step == 0):
                    update_steps.append(eta)
                    _, train_accuracy = self.compute_accuracy(X, Y)
                    _, val_accuracy = self.compute_accuracy(val_X, val_Y)
                    accuracies['train'].append(train_accuracy)
                    accuracies['val'].append(val_accuracy)
                    eta += eta_incr
                    
                t += 1

        return accuracies, losses, costs, update_steps
 

 def compute_grads_num(clf, X, Y, lamda=0, h=1e-5):
    grad_Ws = []
    grad_bs = []
    grad_gammas = []
    grad_betas = []
    number_layers = len(clf.layers)
    
    for layer in clf.layers:
        grad_Ws.append(np.zeros(layer.W.shape))
        grad_bs.append(np.zeros(layer.b.shape))
        if layer.apply_bn:
            grad_gammas.append(np.zeros(layer.gamma.shape))
            grad_betas.append(np.zeros(layer.beta.shape))
    
    
    D = X.shape[0]
    
    for l, layer in enumerate(clf.layers):
        for i in range(len(layer.b)):
            b = np.copy(layer.b)
            b_try = np.array(layer.b)
            b_try[i] -= h
            layer.b = b_try
            _, c1 = clf.compute_cost(X, Y)
            b_try = np.array(b)
            b_try[i] += h
            layer.b = b_try
            _, c2 = clf.compute_cost(X, Y)
            grad_bs[l][i] = (c2-c1) / (2*h)
            layer.b = b
       
    
    for l, layer in enumerate(clf.layers):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                W = np.copy(layer.W)
                W_try = np.array(layer.W)
                W_try[i,j] -= h
                layer.W = W_try
                _, c1 = clf.compute_cost(X, Y)
                W_try = np.array(W)
                W_try[i,j] += h
                layer.W = W_try
                _, c2 = clf.compute_cost(X, Y)
                grad_Ws[l][i,j] = (c2-c1) / (2*h)
                layer.W = W
            
    
    if clf.batch_normalisation:
        for l, layer in enumerate(clf.layers[:-1]):
            for i in range(len(layer.gamma)):
                gamma = np.copy(layer.gamma)
                gamma_try = np.array(layer.gamma)
                gamma_try[i] -= h
                layer.gamma = gamma_try
                _, c1 = clf.compute_cost(X, Y)
                gamma_try = np.array(gamma)
                gamma_try[i] += h
                layer.gamma = gamma_try
                _, c2 = clf.compute_cost(X, Y)
                grad_gammas[l][i] = (c2-c1) / (2*h)
                layer.gamma = gamma
    
        for l, layer in enumerate(clf.layers[:-1]):
            for i in range(len(layer.beta)):
                beta = np.copy(layer.beta)
                beta_try = np.array(layer.beta)
                beta_try[i] -= h
                layer.beta = beta_try
                _, c1 = clf.compute_cost(X, Y)
                beta_try = np.array(beta)
                beta_try[i] += h
                layer.beta = beta_try
                _, c2 = clf.compute_cost(X, Y)
                grad_betas[l][i] = (c2-c1) / (2*h)
                layer.beta = beta

    return grad_Ws, grad_bs, grad_gammas, grad_betas



def check_gradients_are_equal(clf, X_train, Y_train_1hot, num_samples=20):
    # choose random subset of num_samples to test on
    indices = np.arange(X_train.shape[1])
    np.random.shuffle(indices)
    indices = indices[:num_samples]
  
    X_a = np.take(X_train, indices, axis=1)
    Y_a = np.take(Y_train_1hot, indices, axis=1)
    X_n = np.copy(X_a)
    Y_n = np.copy(Y_a)
    
    weights_copies = []
    bias_copies = []
    gammas_copies = []
    betas_copies = []
    for i, layer in enumerate(clf.layers):
        weights_copies.append(np.copy(layer.W))
        bias_copies.append(np.copy(layer.b))
        if layer.apply_bn and i < len(clf.layers)-1:
            gammas_copies.append(np.copy(layer.gamma))
            betas_copies.append(np.copy(layer.beta))
        
    grad_Ws, grad_bs, grad_gammas, grad_betas = clf.compute_gradients(X_a, Y_a)
    
    # reset params
    for i, layer in enumerate(clf.layers):
        layer.W = weights_copies[i]
        layer.b = bias_copies[i]
        
    grad_Ws_num, grad_bs_num, grad_gammas_num, grad_betas_num = compute_grads_num(clf, X_n, Y_n)
    
    def compare_gradients(g_n, g_a, eps=0.000001):
#         print('numerical', g_n)
#         print('analytical', g_a)
        err = np.linalg.norm(g_a-g_n) / max(eps, np.linalg.norm(g_a) + np.linalg.norm(g_n))
        print('relative error', err)
        print('err < 10**-6', err < 10**-6)
        return err

    # check whether they're similar
    err_Ws = []
    err_bs = []
    err_betas = []
    err_gammas = []
    for i, layer in enumerate(clf.layers):
        print('W gradients')
        err_Ws.append(compare_gradients(grad_Ws_num[i], grad_Ws[i]))
        print('b gradients')
        err_bs.append(compare_gradients(grad_bs_num[i], grad_bs[i]))
        
        print(clf.batch_normalisation)
        if clf.batch_normalisation and i < len(clf.layers)-1:
                print('Gamma gradients')
                err_gammas.append(compare_gradients(grad_gammas_num[i], grad_gammas[i]))
                print('Beta gradients')
                err_betas.append(compare_gradients(grad_betas_num[i], grad_betas[i]))
    return err_Ws, err_bs, err_betas, err_gammas
    

def train_model(params, datasets, plot_acc_costs=False):
    train_X, train_Y, val_X, val_Y, test_X , test_Y = datasets
    n_classes = 10
    input_dim = train_X.shape[0]
    random_shuffle = True

    dropouts = params['dropouts'] if 'dropouts' in params else None

    if 'eta' in params:
            clf = KLayerClassifier(batch_size=params['batch_size'], \
                           lamda=params['lamda'], cyclical_lr=False, \
                           eta=params['eta'], \
                           batch_normalisation=params['batch_normalisation'])
    else:
        clf = KLayerClassifier(batch_size=params['batch_size'], n_cycles=params['n_cycles'], \
                           lamda=params['lamda'], cyclical_lr=True, n_s=params['n_s'], \
                           eta_min=params['eta_min'], eta_max=params['eta_max'], \
                           batch_normalisation=params['batch_normalisation'])
    clf.build_model(input_dim, params['hidden_nodes'], dropout_threshes=dropouts)
        
    train_X, val_X, test_X = clf.normalise(train_X, val_X, test_X)
    apply_jitter = params['apply_jitter'] if 'apply_jitter' in params else False
    
    
    if 'n_epochs' in params:
        accuracies, losses, costs, update_steps = clf.train(train_X, train_Y, random_shuffle, n_epochs=params['n_epochs'], get_accuracies_costs=plot_acc_costs, 
                                  val_X = val_X, val_Y = val_Y, apply_jitter=apply_jitter)
    else:
        accuracies, losses, costs, update_steps = clf.train(train_X, train_Y, random_shuffle, get_accuracies_costs=plot_acc_costs, 
                                  val_X = val_X, val_Y = val_Y, apply_jitter=apply_jitter)
        
    if plot_acc_costs:
        plot_costs_accuracies(accuracies, losses, costs, update_steps)

    _, test_accuracy = clf.compute_accuracy(test_X, test_Y) # use lables to compute accuracy
    _, val_accuracy = clf.compute_accuracy(val_X, val_Y)
    print('test_accuracy', test_accuracy)
    return {'test_accuracy': test_accuracy, 'val_accuracy': val_accuracy}


def sensitivity_experiments(params, datasets, plot_acc_costs=False):
    train_X, train_Y, val_X, val_Y, test_X , test_Y = datasets
    n_classes = 10
    input_dim = train_X.shape[0]
    random_shuffle = True

    clf = KLayerClassifier(batch_size=params['batch_size'], n_cycles=params['n_cycles'], \
                           lamda=params['lamda'], cyclical_lr=True, n_s=params['n_s'], \
                           eta_min=params['eta_min'], eta_max=params['eta_max'], \
                           batch_normalisation=params['batch_normalisation'])
    if 'sigma' in params:
        clf.build_model(input_dim, params['hidden_nodes'],  \
                        initialisation=params['initialisation'], sigmas=params['sigma'])
    else:
        clf.build_model(input_dim, params['hidden_nodes'], initialisation=params['initialisation'])
        
    train_X, val_X, test_X = clf.normalise(train_X, val_X, test_X)
 
    
    accuracies, losses, costs, update_steps = clf.train(train_X, train_Y, random_shuffle, get_accuracies_costs=plot_acc_costs, 
                                  val_X = val_X, val_Y = val_Y)
        
    if plot_acc_costs:
        plot_costs_accuracies(accuracies, losses, costs, update_steps)

    _, test_accuracy = clf.compute_accuracy(test_X, test_Y) # use lables to compute accuracy
    _, val_accuracy = clf.compute_accuracy(val_X, val_Y)
    print('test_accuracy', test_accuracy)
    return {'test_accuracy': test_accuracy, 'val_accuracy': val_accuracy}



def display(image):
    ''' takes in image of shape (3072,) '''
    im = np.reshape(image,(3, 32,32))
    im = np.transpose(im, (1,2,0))
    plt.imshow(im)
    plt.show()
    
# Visualise random training image
display(train_X[:,18])
display(train_X[:,19])

def add_jitter(images, display=False, noise_level=0.01):
    D, N = images.shape
    ims = np.reshape(images, (3,32,32,N))
    ims = np.transpose(ims, (1,2,0,3))
    h, w, c, _ = ims.shape # (32,32,3)
    
#     noise = np.random.randint(0,noise_level,(h,w,c,N))
    noise = np.random.uniform(low=0, high=noise_level, size=(h,w,c,N))
    zitter = np.zeros_like(ims)
    zitter[:,:,:,:] = noise
    
    noise_added = ims + zitter

    combined = np.vstack((ims[:int(h/2),:,:,:], noise_added[int(h/2):,:,:,:]))
#     combined = np.where(combined > 255, 255, combined)
    
    
    if display:
        for i in range(N):
            plt.imshow(combined[:,:,:,i], interpolation='none')
            plt.show()
    
    combined_flattened = np.transpose(combined, (2,0,1,3))
    combined_flattened = combined_flattened.reshape(D,N)
    return combined_flattened



def rotate_images(images, labels, display=False):
    D, N = images.shape
    ims = np.reshape(images, (3,32,32,N))
    ims = np.transpose(ims, (1,2,0,3))
    h, w, c, _ = ims.shape # (32,32,3)
    
    rotated = []
    rotations = [90, 180, 270]
    rotated_labels = []
    
    for i in range(N):
        r = np.random.choice(rotations)
        rotated_im = rotate(ims[:,:,:,i], angle=r, resize=True)
        if display:
            plt.imshow(rotated_im, interpolation='none')
            plt.show()

        rotated.append(rotated_im)
        rotated_labels.append(labels[:,i])

    rotated = np.asarray(rotated)
    rotated_labels = np.asarray(rotated_labels).T
    
    rotated = np.transpose(rotated, (2,0,1,3))
    rotated = rotated.reshape(D,N)
    return rotated, rotated_labels




if __name__ == '__main__':
	# load in data for checking gradients
	CIFARDATA = CIFAR10Data(dataset_dir='../datasets/cifar-10-batches-py/')
	train_X, train_Y = CIFARDATA.load_batch('data_batch_1')
	val_X, val_Y = CIFARDATA.load_batch('data_batch_2')
	test_X, test_Y = CIFARDATA.load_batch('test_batch')
	X = train_X[:10,:] # take 10 dimensions
	input_dim = X.shape[0]

	# 2 layer classifier gradients with no batch normalisation
	n_s = 5 * 45000 / 100
	model = KLayerClassifier(batch_size=100, eta=0.005, n_cycles=2, lamda=.005, cyclical_lr=True, n_s=n_s, eta_min=10**-5, eta_max=0.1)
	train_X, val_X, test_X = model.normalise(train_X, val_X, test_X)
	hidden_nodes = [50]
	model.build_model(input_dim, hidden_nodes)
	check_gradients_are_equal(model, X, train_Y, num_samples=100)

	# 3 layer classifier gradients with no batch normalisation
	n_s = 5 * 45000 / 100
	model = KLayerClassifier(batch_size=100, eta=0.005, n_cycles=2, lamda=.005, cyclical_lr=True, n_s=n_s, eta_min=10**-5, eta_max=0.1)
	train_X, val_X, test_X = model.normalise(train_X, val_X, test_X)
	hidden_nodes = [10, 10]
	model.build_model(input_dim, hidden_nodes)
	check_gradients_are_equal(model, X, train_Y, num_samples=100)

	# 4 layer classifier gradients with no batch normalisation
	n_s = 5 * 45000 / 100
	model = KLayerClassifier(batch_size=100, eta=0.005, n_cycles=2, lamda=.005, cyclical_lr=True, n_s=n_s, eta_min=10**-5, eta_max=0.1)
	train_X, val_X, test_X = model.normalise(train_X, val_X, test_X)
	hidden_nodes = [10, 10, 10]
	model.build_model(input_dim, hidden_nodes)
	check_gradients_are_equal(model, X, train_Y, num_samples=100)

	# 4 layer classifier gradients with no batch normalisation
	n_s = 5 * 45000 / 100
	model = KLayerClassifier(batch_size=100, eta=0.005, n_cycles=2, lamda=.005, cyclical_lr=True, n_s=n_s, eta_min=10**-5, eta_max=0.1, batch_normalisation=True)
	train_X, val_X, test_X = model.normalise(train_X, val_X, test_X)
	hidden_nodes = [10, 10, 10]
	model.build_model(input_dim, hidden_nodes)
	err_Ws, err_bs, err_betas, err_gammas = check_gradients_are_equal(model, X, train_Y, num_samples=100)


	# load in all data
	train_X, train_Y = CIFARDATA.load_batches(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'])
	val_X, val_Y = CIFARDATA.load_batch('data_batch_5')
	train_X = np.hstack((train_X, val_X[:,:5000]))
	train_Y = np.hstack((train_Y, val_Y[:,:5000]))
	val_X = val_X[:,5000:]
	val_Y = val_Y[:,5000:]
	test_X, test_Y = CIFARDATA.load_batch('test_batch')
	datasets = [train_X, train_Y, val_X, val_Y, test_X , test_Y]



    # 3-layer network without batch normalisation
    params = {'lamda': 0.005, 'n_cycles':  2, 'batch_size': 100, 'hidden_nodes': [50, 50],
      'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100, 'batch_normalisation': False}
	train_model(params, datasets, plot_acc_costs=True)


	# 9-layer network without batch normalisation
	params = {'lamda': 0.005, 'n_cycles':  2, 'batch_size': 100, 'batch_normalisation': False, \
      'hidden_nodes': [50, 30, 20, 20, 10, 10, 10, 10], \
      'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100}
	train_model(params, datasets, plot_acc_costs=True)


	### Apply Batch Normalisation ###
	params = {'lamda': 0.005, 'n_cycles':  2, 'batch_size': 100, 'hidden_nodes': [50, 50],
      'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100, 'batch_normalisation': True}
	train_model(params, datasets, plot_acc_costs=True)

	# coarse-to-fine search
	test_results = []

	l = -5
	while l <= -1:
	    print(l)
	    params = {'lamda': 10**l, 'l': l, 'n_cycles':  2, 'batch_size': 100, 'hidden_nodes': [50, 50],
	          'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100, 'batch_normalisation': True}
	    res = train_model(params, datasets, plot_acc_costs=False)
	    test_results.append({**params, **res})
	    l += 0.5

	import pandas as pd
	df_coarse_search = pd.DataFrame(test_results).sort_values(by=['test_accuracy', 'val_accuracy'], ascending=False)
	df_coarse_search.to_csv('coarse_search.csv')
	df_coarse_search


	# random search, running in refined interval
	import random

	results = []

	# we use the range for lambda that had the highest accuracy in the coarse search
	l_min = -2.5 # TODO 
	l_max = -2.0

	for x in range(10):
	    l = l_min + (l_max - l_min)*random.uniform(0, 1)
	    params = {'batch_size': 100, 'hidden_nodes': [50, 50], 'n_cycles': 2, 'batch_normalisation': True,
	          'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100, 'lamda': 10**l, 'l': l}
	    res = train_model(params, datasets, plot_acc_costs=False)
	    results.append({**params, **res})
	    
	df_random_search = pd.DataFrame(results).sort_values(by=['test_accuracy', 'val_accuracy'], ascending=False)
	df_random_search.to_csv('random_search.csv')
	df_random_search

	# one more round of refined random search
	import random

	results = []

	# we use the range for lambda that had the highest accuracy in the coarse search
	l_min = -2.449 # TODO 
	l_max = -2.143

	for x in range(20):
	    l = l_min + (l_max - l_min)*random.uniform(0, 1)
	    params = {'batch_size': 100, 'hidden_nodes': [50, 50], 'n_cycles': 2, 'batch_normalisation': True,
	          'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100, 'lamda': 10**l, 'l': l}
	    res = train_model(params, datasets, plot_acc_costs=False)
	    results.append({**params, **res})
	    
	df_random_ref_search = pd.DataFrame(results).sort_values(by=['test_accuracy', 'val_accuracy'], ascending=False)
	df_random_ref_search.to_csv('random_search_refined.csv')
	df_random_ref_search


	### 9-layer network with same parameters as optimised 3-layer network  with batch normalisation ###
	params = {'lamda': 0.006669, 'n_cycles':  3, 'batch_size': 100, 'batch_normalisation': True, \
      'hidden_nodes': [50, 30, 20, 20, 10, 10, 10, 10], \
      'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100}
	train_model(params, datasets, plot_acc_costs=True)	


	### Sensitivity to initialisation ###
	
	# batch norm with normally distributed initialisation
	sigmas = [1e-1, 1e-3, 1e-4 ]
	results = []
	for sigma in sigmas:
	    # train with and without batch normalisation
	    params = {'lamda': 0.005, 'n_cycles':  2, 'batch_size': 100, 'hidden_nodes': [50, 50],
	              'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 2 * 45000 / 100, 'batch_normalisation': True, \
	             'sigmas': sigma, 'initialisation': 'normal'}
	    res = sensitivity_experiments(params, datasets, plot_acc_costs=True)
	    results.append({**params, **res})
	    
	    params = {'lamda': 0.005, 'n_cycles':  2, 'batch_size': 100, 'hidden_nodes': [50, 50],
	              'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 2 * 45000 / 100, 'batch_normalisation': False, \
	             'sigmas': sigma, 'initialisation': 'normal'}
	    res = sensitivity_experiments(params, datasets, plot_acc_costs=True)
	    results.append({**params, **res})


	# plot results 
	df_sensitivity = pd.DataFrame(results).sort_values(by=['test_accuracy', 'val_accuracy'], ascending=False)
	df_sensitivity.to_csv('df_sensitivity.csv')

	for batch_normalisation in [True, False]:
	    df = df_sensitivity[df_sensitivity['batch_normalisation']==batch_normalisation].sort_values(by=['sigmas'], ascending=True)
	    y = list(df['test_accuracy'])
	    x = list(df['sigmas'])
	#     x = np.log10(x)
	    plt.plot(x, y, '-o', label=f'batch_normalisation={batch_normalisation}')

	plt.xlabel('Sigma')
	plt.xscale('log')
	plt.legend()
	plt.ylabel('Test Accuracy')  
	plt.show()


	### Optimising the network ###

	# deeper network
	lamdas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]

	input_dim = train_X.shape[0]

	# results = []
	for i in range(3,12,2): # number of hidden layers
	    for lamda in lamdas:
	        params = {'n_cycles':  1, 'batch_size': 100, 'eta_min': 10**-5, 'eta_max': 0.1, \
	                  'n_s': 5 * 45000 / 100, 'batch_normalisation': True, 'lamda': lamda, \
	                 'hidden_nodes': [50]*i, 'number_layers': i+1}
	        res = train_model(params, datasets, plot_acc_costs=False)
	        print(res['test_accuracy'])
	        results.append({**params, **res})
	   
	import pandas as pd
	# df_exhaustive_search = pd.DataFrame(results).sort_values(by=['test_accuracy', 'val_accuracy'], ascending=False)
	# df_exhaustive_search.to_csv('df_exhaustive_search.csv')
	df_exhaustive_search = pd.read_csv('df_exhaustive_search.csv')
	lamdas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
	for lamda in lamdas:
	    df = df_exhaustive_search[df_exhaustive_search['lamda']==lamda].sort_values(by=['number_layers'], ascending=True)
	    y = list(df['test_accuracy'])
	    x = list(df['number_layers'])
	    plt.plot(x, y, '-o', label=f'lambda={lamda}')
	plt.xlabel('Number of layers')
	plt.legend()
	plt.ylabel('Test Accuracy')  

	# augmenting with jitter
	

	# check performance for more cycles (3) with optimal lambda
	params = {'lamda': 0.006669, 'n_cycles':  1, 'batch_size': 100, 'hidden_nodes': [50, 50],
	          'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100, 'batch_normalisation': True,
	         'apply_jitter': True}
	train_model(params, datasets, plot_acc_costs=True)

	# check performance for more cycles (3) with optimal lambda
	params = {'lamda': 0.006669, 'n_cycles':  3, 'batch_size': 100, 'hidden_nodes': [50, 50],
	          'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100, 'batch_normalisation': True,
	         'apply_jitter': True}
	train_model(params, datasets, plot_acc_costs=True)


	# Augment training set with rotation and noise
	from skimage.transform import rotate

	train_X_a = np.copy(train_X)
	train_Y_a = np.copy(train_Y)

	# rotate images on 10% of data
	indices = np.random.choice([i for i in range(train_X_a.shape[1])], int(0.1*train_X_a.shape[1]), replace=False)
	data_to_rotate = train_X_a[:,indices]
	data_to_rotate_labels = train_Y_a[:,indices]
	rotated_X, rotated_Y = rotate_images(data_to_rotate, data_to_rotate_labels)
	# train_X_a = np.hstack((train_X_a, rotated_X))
	# train_Y_a = np.hstack((train_Y_a, rotated_Y))


	# take 5% of data and add jitter
	indices = np.random.choice([i for i in range(train_X_a.shape[1])], int(0.05*train_X_a.shape[1]), replace=False)
	data_to_jitter = train_X_a[:,indices]
	jitter_labels = train_Y_a[:,indices]
	jitter_data = add_jitter(data_to_jitter)

	# take 50% of rotated data and add jitter
	indices = np.random.choice([i for i in range(rotated_X.shape[1])], int(0.5*rotated_X.shape[1]), replace=False)
	data_to_jitter = rotated_X[:,indices]
	jitter_rotated_labels = rotated_Y[:,indices]
	jitter_rotated_data = add_jitter(data_to_jitter)

	train_X_a = np.hstack((train_X_a, rotated_X, jitter_data, jitter_rotated_data))
	train_Y_a = np.hstack((train_Y_a, rotated_Y, jitter_labels, jitter_rotated_labels))

	# randomly shuffle
	indices = [i for i in range(train_X_a.shape[1])]
	train_X_a = train_X_a[:,indices]
	train_Y_a = train_Y_a[:,indices]

	# new training set sizes
	print(train_X_a.shape)
	print(train_Y_a.shape)

	np.save('augmented_data.npy', train_X_a)
	np.save('augmented_labels.npy', train_Y_a)

	train_X_a = np.load('augmented_data.npy')
	train_Y_a = np.load('augmented_labels.npy')
	# use augmented dataset in training
	params = {'lamda': 0.006669, 'n_cycles':  3, 'batch_size': 100, 'hidden_nodes': [50, 50],
	          'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 54000 / 100, 'batch_normalisation': True}
	train_model(params, datasets=[train_X_a, train_Y_a, val_X, val_Y, test_X , test_Y], plot_acc_costs=True)

	# Applying dropout 
	params = {'lamda': 0.006669, 'n_cycles':  3, 'batch_size': 100, 'hidden_nodes': [50, 50], 'dropouts':[0.05,0.05],
	          'eta_min': 10**-5, 'eta_max': 0.1, 'n_s': 5 * 45000 / 100, 'batch_normalisation': True}
	train_model(params, datasets, plot_acc_costs=True)


