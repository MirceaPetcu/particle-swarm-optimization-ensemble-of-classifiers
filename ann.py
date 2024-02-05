import numpy as np

def ReLU(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1/(1+np.exp(-x))

class ANN:
    def __init__(self, input_size=8, hidden_size=30,hidden_size2=30, output_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        self.W1 = np.random.rand(self.input_size,self.hidden_size) #8x30 --> 240
        self.W2 = np.random.rand(self.hidden_size,self.hidden_size2) #30x30 --> 900
        self.W3 = np.random.rand(self.hidden_size2,self.output_size) #30x2 --> 60 
        
    def forward(self, x):
        x = self.W1.T.dot(x)
        x = ReLU(x)
        x = self.W2.T.dot(x)
        x = ReLU(x)
        x = self.W3.T.dot(x)
        return np.argmax(x)
        
        