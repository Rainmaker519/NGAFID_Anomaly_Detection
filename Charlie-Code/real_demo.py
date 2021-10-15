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
from time import sleep
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
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        return BCE + KLD
    
def loss_fn_2(recon_x, x, mu, logvar):
    tmp = x - recon_x
    tmp = tmp * tmp
    loss = torch.sum(tmp)/len(recon_x)
    return loss
#------------------------------------------------------------------------------------------------------------------------------------- 
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=10, z_dim=1):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential( #this is q (n_flow)
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h_dim, z_dim*2)
            #add pnf here for OmniAnomaly
        )
        
        self.decoder = nn.Sequential( #this is p (n_flow)
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
    
class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs
    #--------------------------------------------------------------------------------------------------------------------------------   
if __name__ == '__main__':
    data = pd.read_csv('c172_file_1.csv')
    step_size= 1
    batch = 128
    index_step_length = 31
    epochs = 40
    #---------------------------------------------------------------------------------------------------------------------------------
    labels, X, Y, XX, YY = Splitting_dataset(data, step_size)
    demo = VAE(index_step_length)
    demo.double()
    optimizer = torch.optim.Adam(demo.parameters(), lr=1e-3)

    idx = 0
    
    anomaly_history = []
    loss_history = []
    avgSum = 0
    avgCount = 0

    for epoch in range(epochs):
        for i in range(len(XX)):
            localX = torch.tensor(XX[i])
            recon, mu, logvar = demo(localX)
            loss = loss_fn(recon, localX, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx = idx + 1
            
            avgSum = avgSum + torch.mean(loss/batch)
            avgCount = avgCount + 1
            
            anomaly_score = torch.mean(localX/recon)
            
            if idx%40 == 0:
                loss_history.append(avgSum/avgCount)
                anomaly_history.append(anomaly_score)
                avgSum = 0
                avgCount = 0
            
            if idx%100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data.item()/batch))
                
            #plt.plot(loss_history,'g-',label='h 10,z 2')
            plt.plot(anomaly_history,'g-',label='h 10,z 2')
    #---------------------------------------------------------------------------------------------------------------------------------
    data = pd.read_csv('c172_file_1.csv')
    labels, X, Y, XX, YY = Splitting_dataset(data, step_size)
    demo = VAE(index_step_length,h_dim=28,z_dim=1)
    demo.double()
    optimizer = torch.optim.Adam(demo.parameters(), lr=1e-3)

    idx = 0
    
    anomaly_history = []
    loss_history = []
    avgSum = 0
    avgCount = 0

    for epoch in range(epochs):
        for i in range(len(XX)):
            localX = torch.tensor(XX[i])
            recon, mu, logvar = demo(localX)
            loss = loss_fn(recon, localX, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx = idx + 1
            
            avgSum = avgSum + torch.mean(loss/batch)
            avgCount = avgCount + 1
            
            anomaly_score = torch.mean(localX/recon)
            
            if idx%40 == 0:
                loss_history.append(avgSum/avgCount)
                anomaly_history.append(anomaly_score)
                avgSum = 0
                avgCount = 0
            
            if idx%100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data.item()/batch))
                
            #plt.plot(loss_history,'r-',label='h 35,z 4')
            plt.plot(anomaly_history,'r-',label='h 35,z 4')
    #---------------------------------------------------------------------------------------------------------------------------------
    data = pd.read_csv('c172_file_1.csv')
    labels, X, Y, XX, YY = Splitting_dataset(data, step_size)
    demo = VAE(index_step_length,h_dim=8,z_dim=3)
    demo.double()
    optimizer = torch.optim.Adam(demo.parameters(), lr=1e-3)

    idx = 0
    
    anomaly_history = []
    loss_history = []
    avgSum = 0
    avgCount = 0

    for epoch in range(epochs):
        for i in range(len(XX)):
            localX = torch.tensor(XX[i])
            recon, mu, logvar = demo(localX)
            loss = loss_fn(recon, localX, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx = idx + 1
            
            avgSum = avgSum + torch.mean(loss/batch)
            avgCount = avgCount + 1
            
            anomaly_score = torch.mean(localX/recon)
            
            if idx%40 == 0:
                loss_history.append(avgSum/avgCount)
                anomaly_history.append(anomaly_score)
                avgSum = 0
                avgCount = 0
            
            if idx%100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data.item()/batch))
                
            #plt.plot(loss_history,'b-',label='h 8,z 2')
            plt.plot(anomaly_history,'b-',label='h 8,z 2')
    #---------------------------------------------------------------------------------------------------------------------------------
    print("Using VAE cross entropy loss function")