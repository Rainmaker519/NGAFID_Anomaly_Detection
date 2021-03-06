import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pyreadr as py
import gc
import torch
from torch import nn
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from torch.utils.data import Dataset
  
    
BASE_SEQ_LEN = 500
NUM_VAR = 52
INPUT_SEQ_LEN = 32 
OUTPUT_SEQ_LEN = 1
BATCH_SIZE = 128
    
class VAE(nn.Module):
    def __init__(self, image_size=54, h_dim=20, z_dim=1):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim)
            #add pnf here for OmniAnomaly
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
class TEPDataset(Dataset):
    def __init__():
        print("")
    def __len__():
        print("")
    def __get_item__():
        print("")
    
    
#Split and reshape the data set by step_size , use min-max or stanrdardlize method to rescale the data
def split_dataset(data, step_size, scale=True, scaler_type=MinMaxScaler):
        l = len(data) 
        data = scaler_type().fit_transform(data)
        Xs = []
        Ys = []
        for i in range(0, (len(data) - step_size)):
            Xs.append(data[i:i+step_size])
            Ys.append(data[i:i+step_size])
        train_x, test_x, train_y, test_y = [np.array(x) for x in train_test_split(Xs, Ys)]
        assert train_x.shape[2] == test_x.shape[2] == (data.shape[1] if (type(data) == np.ndarray) else len(data))
        return  (train_x.shape[2], train_x, train_y, test_x, test_y)

   
def summary(model):
    print(model.summary())

    
def trainloss(model,data_X):
    x_train_pred = model.predict(data_X)
    train_loss = np.square(np.mean(np.abs(x_train_pred - data_X), axis=1))

def testloss(model, data_XX):
    x_test_pred = model.predict(data_XX)
    test_loss = np.square(np.mean(np.abs(x_test_pred - data_XX),axis=1))

    
def get_threshold(train_loss):
    threshold = np.max(train_loss)
    print (threshold)
    

def df_to_np_array(df): 
    examples = [] 
    for i in df.simulationRun.unique():
        examples.append(df[df.simulationRun == i].iloc[:, 3:].values)
    
    return np.stack(examples)


def get_slice_random_segment(length):
    def slice_random_segment(x):
        start = tf.random.uniform(shape=[], minval=0, maxval= BASE_SEQ_LEN-length, dtype=tf.int64)
        x = x[start:start+length]
        x = tf.reshape(x, (length, NUM_VAR))
        return x 
    return slice_random_segment

def get_split_xy(y_length): 
    def split_xy(x):
        y = x[-y_length:]
        return x[:-y_length], y 
    return split_xy

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    return BCE + KLD

def get_dataset(train, shuffle = False, repeat = False): 
    train_target = torch.tensor(train['faultNumber'].values.astype(np.float32))
    train = torch.tensor(train.drop('faultNumber', axis = 1).values.astype(np.float32)) 
    train_tensor = torch.utils.data.TensorDataset(train, train_target) 
    
    return train_tensor
    

#

#

if __name__ == '__main__':
    #Loading TEP
    base_loc = "C:/Users/Charlie/Desktop/archive/"
    
    a1 = py.read_r(base_loc + "TEP_FaultFree_Training.RData")
    fault_free_training = a1['fault_free_training']
    a1 = None 
    a2 = py.read_r(base_loc + "TEP_Faulty_Training.RData")
    faulty_training = a2['faulty_training']
    a2 = None
    a3 = py.read_r(base_loc + "TEP_FaultFree_Testing.RData")
    fault_free_testing = a3['fault_free_testing']
    a3 = None 
    a4 = py.read_r(base_loc + "TEP_Faulty_Testing.RData")
    faulty_testing = a4['faulty_testing']
    a4 = None 
    
    gc.collect()
    
    for col in fault_free_training.columns: 
        fault_free_training.loc[: , col] = fault_free_training.loc[: , col].astype('float32')

    for col in faulty_training.columns: 
        faulty_training.loc[: , col] = faulty_training.loc[: , col].astype('float32')

    for col in fault_free_testing.columns: 
        fault_free_testing.loc[: , col] = fault_free_testing.loc[: , col].astype('float32')

    for col in faulty_testing.columns: 
        faulty_testing.loc[: , col] = faulty_testing.loc[: , col].astype('float32')
    #
    
    
    train_df = fault_free_training
    
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_df.iloc[:, 3:].values)

    #train_df.iloc[:, 3:] = scaler.transform(train_df.iloc[:, 3:].values)
    
    train_ds = get_dataset(train_df, shuffle = True, repeat=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VAE()
    if torch.cuda.is_available():
        model.cuda()
    
    print("DataSet Length: " + str(len(train_ds)))
    print(train_ds)
    
    data_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    #data_loader.to(device)
    print("DataLoader Length: " + str(len(data_loader)))
    epochs = 10

    for epoch in range(epochs):
        for local_batch, local_labels in data_loader:
            #locX,locY = i
            #locX, locY = torch.Tensor(locX), torch.Tensor(locY)
            local_batch.to(device)
            local_labels.to(device)

            recon_x, mu, logvar = model(local_batch)
            loss = loss_fn(recon_x,local_batch,mu,logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx%100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data.item()/bs))
                recon_x, _, _ = model(fixed_x)


    