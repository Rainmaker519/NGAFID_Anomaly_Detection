import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
#--------------------------------------------------------------------------------------------------------------------------------------
#Split and reshape the data set by step_size , use min-max or stanrdardlize method to rescale the data
def Splitting_dataset(data, step_size, scale=True, scaler_type=MinMaxScaler):
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

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def loss_fn(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        return BCE + KLD
    
def loss_fn_2(recon_x, x, mu, logvar):
    losses = 0
    for i in range(len(recon_x)):
        tmp = x[i]-recon_x[i]
        loss = loss + (tmp*tmp)
    return loss
#------------------------------------------------------------------------------------------------------------------------------------- 
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=27, z_dim=13):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
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
    
"""    
class TestModel(nn.Module):
    def __init__(self,input_size=54):
        super(TestModel,self).__init__()
        self.model = nn.Sequential(
            nn.RNN(input_size,input_size)
            nn.LeakyReLU(0.2)
            nn.RNN(input_size,input_size)
            nn.Dropout2d(.5)
            nn.Linear(input_size,5)
            nn.Linear(5,input_size))
        def forward(self,next):
            return model(next)
"""            
#-------------------------------------------------------------------------------------------------------------------------------------   
if __name__ == '__main__':
    data = pd.read_csv('c172_file_1.csv')
    step_size= 10
    #epoch = 1000
    batch = 128
    index_step_length = 31
    loss_type = "mse"
    optimizer_type = "adam"
    #recon_x
    labels, X, Y, XX, YY = Splitting_dataset(data, step_size)
    demo = VAE(index_step_length)
    demo.double()
    optimizer = torch.optim.Adam(demo.parameters(), lr=1e-2)
    
    epochs = 10
    
    idx = 0
    
    loss_history = []

    for epoch in range(epochs):
        for i in range(len(XX)):
            localX = torch.tensor(XX[i])
            recon, mu, logvar = demo(localX)
            loss = loss_fn(recon, localX, mu, logvar)
            loss_history.append(loss/batch)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx = idx + 1
            
            if idx%100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data.item()/batch))
                
                #recon,_,_ = demo(fixed_x)
                
    plt.plot(loss_history)
    