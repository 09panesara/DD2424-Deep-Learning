import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import copy
import seaborn as sn
sn.set()

from sklearn import preprocessing

class OneHotEncode:
    def __init__(self, chars):
        ''' takes in list set of all characters in test'''
        self.K = len(chars)
        print('dimensionality of chars', self.K)
        self.chars = chars
        self.one_hot_encoder = preprocessing.LabelBinarizer()
        self.one_hot_encoder.fit(chars)
    
    
    def one_hot_encode(self, seq):
        ''' 
        Inputs:
            seq: list of characters to one hot encode
        Returns:
            Y: (output_size, T) where output_size is dimensionality of one hot encoding, T is len(seq)
        '''
        seq_list = []
        seq_list[:0] = seq 
        Y = self.one_hot_encoder.transform(seq_list).T
        assert Y.shape == (self.K, len(seq))
        return Y
    
    def one_hot_to_characters(self, Y):
        ''' 
        Inputs:
            Y: Array of one hot encodings (output size, T)
        Returns:
             and converts back into characters '''
        assert Y.shape[0] == self.K
        chars_list = self.one_hot_encoder.inverse_transform(Y.T)
        seq = ''.join(chars_list)
        return seq
    
    
    
class RNN:
    def __init__(self, chars, hidden_states=100, eta=0.1, seq_length=25):
        '''
        Inputs:
            chars: set of characters in text
            number_states : dimensionality of hidden state
            eta: learning rate
            seq_length: length of sequences used in training 
        Initialises weight matrices and params
        '''
        sigma = 0.01
        self.output_dim = len(chars)
        self.hidden_states = hidden_states #  number of hidden states

        self.U = np.random.randn(hidden_states, self.output_dim)*sigma # applied to input x_t (input-to-hidden connection)
        self.W = np.random.randn(hidden_states, hidden_states)*sigma # applied to prev hidden state h_t-1 (hidden-to-hidden connection)
        self.V = np.random.randn(self.output_dim, hidden_states)*sigma # applied to h_t (hidden-to-output connection)
        self.eta = eta    
        self.b = np.zeros(hidden_states)
        self.c = np.zeros(self.output_dim)
        
        self.batch_size = seq_length
        self.smooth_loss = None
        self.grads = {'U': 0, 'W': 0, 'V': 0, 'b': 0, 'c': 0}
        self.ada_grad_params = {'U': 0, 'W': 0, 'V': 0, 'b': 0, 'c': 0} # m's
        
        OneHotEnc = OneHotEncode(chars)
        self.chars = chars
       
        self.one_hot_encode = OneHotEnc.one_hot_encode
        self.one_hot_to_characters = OneHotEnc.one_hot_to_characters
        
        print('Using', hidden_states, 'hidden states')
        
        
    

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=0)
    
    

    def loss(self, Y, P):
        '''
        Inputs:
            Y: targets (T,K)
            P: probabilities (T,K)
        Returns:
            Cross-entropy loss:    L = -sum_over_t(log(y _tp_t))
        '''
        l = (P * Y).sum(axis=1)
        l[l == 0] = np.finfo(float).eps
        return - np.log(l).sum()
    
            
    def predict_prob(self, X_t, h_tm1):
        # checked
        a_t = self.W @ h_tm1 + self.U @ X_t + self.b
        h_t = np.tanh(a_t)
        o_t = self.V @ h_t + self.c
        p_t = self.softmax(o_t)
        return p_t, h_t
    
    
    def forward_pass(self, X, Y, h_tm1):
        # checked
        '''
        Completes a forward pass of RNN network
        Inputs:
            X: Sequence (self.output_size, T)
            Y: Sequence (self.output_size, T)
            h_tm1: previous state
        Returns:
            loss: cross-entropy loss of sequence
            P: probabilities across timesteps
            H: hidden states across timesteps 
        '''
        T = X.shape[1] # length of sequence
        P = np.zeros((T, self.output_dim))
        H = np.zeros((T+1, self.hidden_states))
        # H[t] = h_t-1
                
        for t in range(T):
            H[t] = h_tm1
            # store in P, H for use in backward algorithm
            P[t], h_tm1 = self.predict_prob(X[:,t], h_tm1)
            
        H[T] = h_tm1
        
        loss = self.loss(Y.T, P)
    
        return loss, P, H
    
    
    def backward_pass(self, X, Y, P, H):
        '''
        Completes a backward pass of RNN network and computes updates of gradients 
        Inputs:
            X: Sequence (self.output_size, T)
            Y: Sequence (self.output_size, T)
            P: probabilities across timesteps of each output character (T, self.output_size)
            H: hidden states across timesteps (T+1, self.number_states)
        Returns:
            grads_W
        '''
        T = X.shape[1]
        self.grads['W'] = np.zeros(self.W.shape)
        self.grads['U'] = np.zeros(self.U.shape)
        self.grads['b'] = np.zeros(self.b.shape)
        dL_da = np.zeros(self.hidden_states)
        
        assert P.shape == Y.T.shape
        dL_do = -(Y.T - P) 

        self.grads['c'] = dL_do.sum(axis=0)
    
        
        # first pass
        self.grads['V'] = np.zeros(self.V.shape)
        for t in range(T):
            self.grads['V'] += np.outer(dL_do[t], H[t+1]) # h_t
        
        # go back in time 
        for t in range(T-1, -1, -1):
            # dL_do[t] is row vector of length K, self.V is matrix (K,m)
            dL_dh = dL_do[t]@self.V + dL_da@self.W # just dl_do_t V for T 
            dL_da = dL_dh * (1 - H[t + 1] ** 2) # H[t+1] = h_t
            self.grads['W'] += np.outer(dL_da, H[t]) # H[t] = h_t-1
           
            self.grads['U'] += np.outer(dL_da, X[:,t]) 
            self.grads['b'] += dL_da
            
       
    
    def compute_gradients(self, X, Y, h_tm1):
        ''' computes gradients of the cost function wrt W and b for batch X, returns updated h_tm1'''

        # forward pass, apply dropout if its set
        loss, P, H = self.forward_pass(X, Y, h_tm1)
        
        # backward pass
        self.backward_pass(X, Y, P, H)
        return H[-1], loss

        
    def update_gradients(self, clip=5):
        ''' appplies AdaGrad update step to all parameters of RNN '''
        weights = {'U': self.U, 'W': self.W, 'V': self.V, 'b': self.b, 'c': self.c}

        for theta in self.ada_grad_params:
            grad = np.clip(self.grads[theta], -clip, +clip) # gradient clipping
            self.ada_grad_params[theta] += grad ** 2
            weights[theta] -= (self.eta * \
                               self.grads[theta] / np.sqrt(self.ada_grad_params[theta] + np.finfo(float).eps))
       
        assert np.array_equal(self.U, weights['U'])
        assert np.array_equal(self.V, weights['V'])
        assert np.array_equal(self.W, weights['W'])
        assert np.array_equal(self.b, weights['b'])
        assert np.array_equal(self.c, weights['c'])
       
    
    def synthesise_sequence(self, x0, h0, T):
        ''' 
        Inputs:
            x0: vector for first (dummy) input to RNN
            h0: hidden state at time 0
        Returns:
            synthesises sequence of length T using current parameters
        '''
        
        seq = np.empty((T, self.output_dim))
        
        h_tm1 = np.copy(h0)
        
        x_t = np.copy(x0)
        
     
        for t in range(T):
#             print('h_tm1', h_tm1)
            # generate next input x_t from current x
            seq[t] = x_t.reshape(self.output_dim)
            # probability for each possible character
            p_t, h_t = self.predict_prob(x_t, h_tm1)   

            
            c_index = np.random.choice(range(len(p_t)), p=p_t)

#             # update params for next iteration
            h_tm1 = h_t
#             x_t = self.one_hot_encode(c).reshape(self.output_dim)
            x_t = np.zeros(x0.shape)
            x_t[c_index] = 1

            seq[t] = x_t

       
        return seq
    
    
    def train(self, X, Y, n_epochs=1):
        '''
        X: K, N
        Y: K, N
        n_epochs: number of epochs to run it for
        '''
        n = X.shape[1]
        
        number_of_batches = int(n / self.batch_size)
        
        assert number_of_batches > 0
        indices = np.arange(X.shape[1])
       
        print('total epochs', n_epochs)
        update_step = 0
        
        update_steps = []
        smoothed_losses = []
        
        for epoch in range(n_epochs):  
            print('epoch', epoch)
            h_tm1 = np.zeros(self.hidden_states) # TODO initialise to zeros for start of epoch
            print(number_of_batches)
            for j in range(number_of_batches):
                
                j_start = j * self.batch_size
                j_end = (j+1) * self.batch_size
                Xbatch = X[:, j_start:j_end]
                Ybatch = Y[:, j_start:j_end]                    
    
                
                # Perform MiniBatch Gradient Descent
                h_tm1, loss = self.compute_gradients(Xbatch, Ybatch, h_tm1)
                
                '''loss will vary a lot from one sequence to the next due to SGD, 
                so keep track of smoothed loss over iterations 
                as weighted sum of smoothed loss and current loss'''
                if self.smooth_loss is None:
                    self.smooth_loss = loss
                else:
                    self.smooth_loss = 0.999 * self.smooth_loss + 0.001 * loss

                if update_step % 500 == 0:
                    print(f'iter={update_step}, smooth_loss={self.smooth_loss}')
                    update_steps.append(update_step)
                    smoothed_losses.append(self.smooth_loss)
                    
                    # check what's happening with the RNN, can we generate sensible text?
                    # pass in first character and current h_t-1 
                if update_step % 500 == 0:
                    x_0 = np.zeros(self.output_dim)
                    x_0[np.random.randint(0, self.output_dim)] = 1
                    
                    h_0 = np.zeros(self.hidden_states)
                    seq = self.synthesise_sequence(x_0, h_0, 200)
#                     seq = self.synthesise_sequence(X[:,0], h_tm1, 100)
                    print(self.one_hot_to_characters(seq.T))
                    
                self.update_gradients() 
                update_step += 1

        
        plt.plot(update_steps, smoothed_losses)
        plt.title('Smoothed Loss')
        plt.xlabel('Update step')
        plt.ylabel('Loss')
        

    def compute_gradients_num(self, X, Y, h_tm1, h, n_comps=20):
        """Numerically computes the gradients of the weight and bias parameters
        Args:
            inputs      (list): indices of the chars of the input sequence
            targets     (list): indices of the chars of the target sequence
            hprev (np.ndarray): previous learnt hidden state sequence
            h     (np.float64): marginal offset
            num_comps    (int): number of entries per gradient to compute
        Returns:
            grads (dict): the numerically computed gradients dU, dW, dV,
                          db and dc
        """
        weights = {"W": self.W, "U": self.U, "V": self.V, "b": self.b, "c": self.c}
        copy_weights = {key: np.copy(weights[key]) for key in weights}
        grads_num  = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U),
                      "V": np.zeros_like(self.V), "b": np.zeros_like(self.b),
                      "c": np.zeros_like(self.c)}

        for theta in weights:
            for i in range(n_comps):
                par_copy = np.copy(weights[theta].flat[i]) # store old parameter
               
                weights[theta].flat[i] = par_copy + h
               
                _, loss1 = self.compute_gradients(X, Y, h_tm1)
                weights[theta].flat[i] = par_copy - h
              
                _, loss2 = self.compute_gradients(X, Y, h_tm1)
                
                weights[theta].flat[i] = par_copy # reset parameter to old value
                grads_num[theta].flat[i] = (loss1 - loss2) / (2*h)

        return grads_num
    


    def check_gradients(self, X, Y, n_epochs=5, n_comps=20):
        """Check similarity between the analytical and numerical gradients
        Args:
            X      (list): indices of the chars of the input sequence
            Y     (list): indices of the chars of the target sequence
            n_epochs: number of epochs to check gradients for
            n_comps    (int): number of gradient comparisons per parameter (i.e. # elements in matrix/vector)
        """
        n = X.shape[1]
        
        number_of_batches = int(n / self.batch_size)
        print('number of batches', number_of_batches)

        assert number_of_batches > 0
        indices = np.arange(X.shape[1])

        print('total epochs', n_epochs)

        def compare_gradients(g_n, g_a, eps=0.000001, n_comps=20):
#             print('numerical', g_n.flat[:n_comps])
#             print('analytical', g_a.flat[:n_comps])
            g_n_copy = np.copy(g_n)
            g_a_copy = np.copy(g_a)
            g_n_copy.flat[n_comps:] = 0
            g_a_copy.flat[n_comps:] = 0
            err = np.linalg.norm(g_a_copy-g_n_copy) / max(eps, np.linalg.norm(g_a_copy) + np.linalg.norm(g_n_copy))
            print('relative error', err)
            print('err < 10**-6', err < 10**-6)
            return err
        
       
       
        for epoch in range(n_epochs):  
            print('epoch', epoch)
            h_tm1 = np.zeros(self.hidden_states) # TODO initialise to zeros for start of epoch
            for j in range(number_of_batches):
                j_start = j * self.batch_size
                j_end = (j+1) * self.batch_size
                Xbatch = X[:, j_start:j_end]
                Ybatch = Y[:, j_start:j_end]                    
                
                # copy old params
                weights = {"W": np.copy(self.W), "U": np.copy(self.U), "V": np.copy(self.V),\
                           "b": np.copy(self.b), "c": np.copy(self.c)}
          

                # Perform MiniBatch Gradient Descent
                h, loss = self.compute_gradients(Xbatch, Ybatch, h_tm1)
               
                grads_a = copy.deepcopy(self.grads)
                
                grads_n = self.compute_gradients_num(Xbatch, Ybatch, h_tm1, h=1e-5, n_comps=n_comps)
                
                    
                print("Checking gradients")
                errs = []
                params = []
                for theta in grads_a:
#                     grads_n[theta].flat[20:] = grads_a[theta].flat[n_comps:]
                    params.append(theta)
                    errs.append(compare_gradients(grads_n[theta], grads_a[theta], n_comps=n_comps))
        
                h_tm1 = h
                # update weights to old weights 
                self.W = np.copy(weights['W'])
                self.U = np.copy(weights['U'])
                self.V = np.copy(weights['V'])
                self.b = np.copy(weights['b'])
                self.c = np.copy(weights['c'])
                
                
                plt.plot(np.arange(5), errs)
                plt.xticks(np.arange(5), labels=params)
                plt.title('Relative error for params between analytical gradient and numerical gradient', y=1.05)
                plt.show()
                
                

                self.update_gradients() 
            
    
if __name__ == '__main__':  
	####### loading data #######
	book_data = open('goblet_book.txt','r').read()
	book_chars = list(set(book_data))


	####### checking gradients #######
	seq_length = 25
	X_chars = book_data[:seq_length]
	# label for an input character is the next character in the book
	Y_chars = book_data[1:seq_length+1]

	rnn = RNN(book_chars)
	X = rnn.one_hot_encode(X_chars) # (K, seq_length)
	Y = rnn.one_hot_encode(Y_chars) # (K, seq_length)

	rnn.check_gradients(X, Y, n_epochs=10, n_comps=40)



	####### Train your RNN using AdaGrad #######
	seq_length = len(book_data)-1
	X_chars = book_data
	# label for an input character is the next character in the book
	Y_chars = book_data[1:seq_length+1]


	rnn = RNN(book_chars)
	X = rnn.one_hot_encode(X_chars) # (K, seq_length)
	Y = rnn.one_hot_encode(Y_chars) # (K, seq_length)
	# print(rnn.chars)

	rnn.train(X, Y, n_epochs=3)



	####### synthesise over longer text #######
	x_0 = np.zeros(rnn.output_dim)
	x_0[np.random.randint(0, rnn.output_dim)] = 1

	h_0 = np.zeros(rnn.hidden_states)
	seq = rnn.synthesise_sequence(x_0, h_0, 1000)
	print(rnn.one_hot_to_characters(seq.T))




