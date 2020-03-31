from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt

class CIFAR10Data():
    def __init__(self, dataset_dir):
        self.batches = {}
        self.label_1hot_encoder = preprocessing.LabelBinarizer()
        self.label_1hot_encoder.fit([x for x in range(10)])
        self.dataset_dir = dataset_dir
        
    def one_hot_encode(self, labels):
        return self.label_1hot_encoder.transform(labels).T
    

    def load_batch(self, batch_name):
        """Loads a data batch
        Args:
            filename (str): filename of the data batch to be loaded
        Returns:
            X (np.ndarray): data matrix (D, N)
            Y (np.ndarray): one hot encoding of the labels (N,)
        """
        filename = self.dataset_dir + batch_name
        with open(filename, 'rb') as f:
            dataDict = pickle.load(f, encoding='bytes')

            X = (dataDict[b"data"]).T
            y = dataDict[b"labels"]
            Y = self.one_hot_encode(y)

        return X, Y
    
    def load_batches(self, batch_names):
        batches = [self.load_batch(batch) for batch in batch_names]
        # check should be hstack or vstack - check shapes
        X = np.hstack([batch[0] for batch in batches])
        Y = np.hstack([batch[1] for batch in batches])
        return X, Y


def montage(W):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(2):
        for j in range(5):
            im  = W[5*i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title(labels[5*i+j])
            ax[i][j].axis('off')
    plt.show()
    

# plot cost function and accuracy
def plot_costs_accuracies(accuracies, costs, epoch_jump=1):
    plt.subplot(1,2,1)
    epochs = np.arange(0,len(costs['train']))*epoch_jump
    plt.plot(epochs, costs['train'], 'g-', label='Train')
    plt.plot(epochs, costs['val'], 'r-', label='Validation')
    plt.title('Cost function')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid('on')

    plt.subplot(1,2,2)
    plt.plot(epochs, accuracies['train'], 'g-', label='Train')
    plt.plot(epochs, accuracies['val'], 'r-', label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid('on')
    
    plt.show()