import numpy as np

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

class RNN:
    def __init__(self,input_size=(30,4), hidden_size=30, output_size=1):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W_hx = np.random.randn(hidden_size, input_size[1])
        self.W_hh = np.random.randn(hidden_size, hidden_size)
        self.W_hy = np.random.randn( output_size,hidden_size)

        # self.b_h = np.zeros((hidden_size, 1))
        # self.b_y = np.zeros((output_size, 1))

        # Initialize hidden state
        self.ht = np.zeros((hidden_size, 1))

    def forward(self, x):
        # Lists to store intermediate values during forward pass
        # self.x, self.h_tilde, self.h, self.y_tilde, self.y = [], [], [], [], []
        assert x.shape == self.input_size, f"Input shape must be {self.input_size} found {x.shape}"
        # Iterate over each time step
        for index, xt in x.iterrows():
              # Input at time step t
            xt = xt.values.reshape(-1,1)
            hx = self.W_hx.dot(xt)  # Weighted input to hidden layer
            # hx = (30,5)x(5,1) = (30,1)
            self.ht = tanh(self.W_hh.dot(self.ht) + hx) # Weighted input + previous hidden state for the new hidden state
            # ht = (30,30)x(30,1) + (30,1) = (30,1)
            yt = self.W_hy.dot(self.ht) # Weighted input state for the output layer (for this timestep) (linear for regression)
            # yt = (1,30)x(30,1) = (1,1)
        return yt.squeeze()

