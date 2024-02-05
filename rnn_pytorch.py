import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as data_utils
import numpy as np

class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_size = 4
        self.hidden_size = 30
        self.output_size = 1
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True,bias=False)
        self.h = torch.zeros((1,1,self.hidden_size))
        
    def forward(self,x):
        x, h = self.rnn(x,self.h)
        x = x[:,-1,:]
        return x.squeeze()[-1]
    
if __name__ == '__main__':
    rnn = RNN()
    df_train = pd.read_csv('apple_train.csv')
    df_val = pd.read_csv('apple_val.csv')
    df_test = pd.read_csv('apple_test.csv') 
    
    df_test = pd.concat([df_val,df_test])
     
    def create_dataset(dataset, lookback):
        X, y = [], []
        import copy
        df_copy = copy.deepcopy(dataset)
        for_target = df_copy['close']
        for_target = for_target[lookback-1:]
        for i in range(lookback-1,len(df_copy)):
            inpts = df_copy.drop(columns=['close'])
            feature = inpts[i-lookback+1:i+1]
            X.append(feature)
            
        return torch.tensor(np.array(X),dtype=torch.float32), torch.tensor(np.array(for_target),dtype=torch.float32)
    
    x_train,y_train = create_dataset(df_train,30)
    
    dataloader = data_utils.DataLoader(data_utils.TensorDataset(x_train,y_train),batch_size=1,shuffle=False)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(rnn.parameters())
    rnn.train()
    import tqdm
    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for x,y in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            y = y.squeeze()
            outputs = rnn(x)
            loss = criterion(outputs,y)
            loss.backward()
            optimizer.step()
    
    
    rnn.eval()
    y_pred = []
    window_size = 30
    x_test,y_test = create_dataset(df_test,window_size) 
    test_dataloader = data_utils.DataLoader(data_utils.TensorDataset(x_test,y_test),batch_size=1,shuffle=False)
    for x,y in test_dataloader:
        pred = rnn(x)
        y_pred.append(pred.detach().numpy())
    y_test = y_test.detach().numpy()
    y_pred = np.array(y_pred)

    from sklearn import metrics
    with open('rnn_performance.txt','a') as f:
        f.write('Pytorch RNN\n')
        f.write(f'Trained for {epochs} epochs\n')
        mae = metrics.mean_absolute_error(y_test,y_pred)
        f.write(f'Pytorch MAE: {mae}\n')
             
    
    