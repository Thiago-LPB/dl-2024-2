import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.loss_history = []
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def predict_proba(self, X):
        yPred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return yPred
    
    def predict(self, X):
        return np.argmax(predict_proba(X), axis=1)
    
    def train(self, X, y):

        self.weights = np.zeros(X.shape[1])
        
        for _ in range(self.epochs):

            for i, x in enumerate(X):
                yPred = self.sigmoid(np.dot(X, self.weights) + self.bias)
    
                dw = (1/X.shape[0]) * np.dot(X.T, (yPred - y))
                db = (1/X.shape[0]) * np.sum(yPred - y)

                self.loss_history.append((-1/X.shape[0])*(np.dot(y,np.log(yPred)) + np.dot(1-y, np.log(1-yPred)))) #Entropia Cruzada
                
                self.weights -= self.lr*dw
                self.bias -= self.lr*db


def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
    return X, y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Logistic Regression Output')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

def plot_loss(model):
    plt.plot(model.loss_history, 'k.')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Training Iterations')
    plt.show()

# Generate dataset
X, y = generate_dataset()

# Train the model
neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
neuron.train(X, y)

# Plot decision boundary
plot_decision_boundary(neuron, X, y)

# Plot loss over training iterations
plot_loss(neuron)
