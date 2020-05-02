from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
import numpy as np 


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

    

# plot cost function and accuracy
def plot_costs_accuracies(accuracies, losses, costs, update_steps):
    fig, ax = plt.subplots(1, 3, figsize=(20,4))
    plt.subplot(1,3,1)
    plt.plot(update_steps, losses['train'], 'g-', label='Train')
    if 'val' in losses:
        plt.plot(update_steps, losses['val'], 'r-', label='Validation')
    plt.title('Loss')
    plt.xlabel('Update step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('on')

    # cost is loss + regularisation
    plt.subplot(1,3,2)
    plt.plot(update_steps, costs['train'], 'g-', label='Train')
    if 'val' in costs:
        plt.plot(update_steps, costs['val'], 'r-', label='Validation')
    plt.title('Cost function')
    plt.xlabel('Update step')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid('on')

    plt.subplot(1,3,3)
    plt.plot(update_steps, accuracies['train'], 'g-', label='Train')
    if 'val' in accuracies:
        plt.plot(update_steps, accuracies['val'], 'r-', label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Update step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid('on')

    plt.show()