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
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
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
    loss = 0

    tmp = x - recon_x
    tmp = tmp * tmp
    loss = torch.sum(tmp)/len(recon_x)
    return loss
#------------------------------------------------------------------------------------------------------------------------------------- 
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=27, z_dim=1):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h_dim, z_dim*2) #is it saying its getting a mu and a var for each z dim out?
            
            #how can I represent the encoder as a distribution acting as the prior?
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )
        
        self.nf = nn.ModuleList(
            [AffineConstantFlow(1),
            AffineConstantFlow(1),
            AffineConstantFlow(1),
            AffineConstantFlow(1)]
        )
        
        #should i add a flow model as a paremeter of the vae for easy access in parameterize?
    
    def reparameterize(self, mu, logvar):
        #print("logvar: " + str(logvar))
        std = logvar.mul(0.5).exp_() 
        #print("std: " + str(std))
        
        #print("test: " + str(to_var(torch.Tensor([.5,1]))))
        #print("test: " + str(torch.Tensor([.5,1])))
        
        esp = to_var(torch.randn(*mu.size()))
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print("esp: " + str(esp))
        #randn - Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
        #esp always ends up being just 1 value, we're getting z sampled from a normal distribution here
        z = mu + std * esp
        #print(z)
        #z is a single sampled point from the distribution created from the mu and var from the encoder output
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        #print("h: " + str(h))
        mu, logvar = torch.chunk(h, 2, dim=1)
        #reparameterize is doing the sampling with esp
        z = self.reparameterize(mu, logvar)
        #print("z: " + str(z))

        zTransformed = self.nf(z) #this is where I apply the normalizing flow once its implemented, the points are transformed as theyre generated
        
        #print("Z: " + str(z))
        #print("ZT: " + str(zTransformed))
        z = zTransformed[0][0]
        #print(z[0][0])
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
            #h = demo.encoder(x)
            #mu, logvar = torch.chunk(h, 2, dim=1)
            x, ld = flow.forward(x)#,demo.reparameterize(mu,logvar))
            log_det += ld
            zs.append(x)
            #here we're just summing the log deterimant of each of the flows
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
        xs, _ = self.flow.backward(z)#should i update this to be the approximate posterior??????????????????
        return xs
    

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of planar flow.
        Args:
            dim: input dimensionality.
        """
        super(PlanarFlow, self).__init__()

        self.linear_u = nn.Linear(dim, dim)
        self.linear_w = nn.Linear(dim, dim)
        self.linear_b = nn.Linear(dim, 1)

    def forward(self, x, v):
        """Forward pass.
        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        u, w, b = self.linear_u(v), self.linear_w(v), self.linear_b(v)

        def m(x):
            return F.softplus(x) - 1.
        def h(x):
            return torch.tanh(x)
        def h_prime(x):
            return 1. - h(x)**2

        inner = (w * u).sum(dim=1, keepdim=True)
        u = u + (m(inner) - inner) * w / (w * w).sum(dim=1, keepdim=True)
        activation = (w * x).sum(dim=1, keepdim=True) + b
        x = x + u * h(activation)
        psi = h_prime(activation) * w
        log_det = torch.log(torch.abs(1. + (u * psi).sum(dim=1, keepdim=True)))

        return x, v, log_det
    
    #def backward(self,
    
#class PNCustomFlow(nn.Module):
    #def __init__(self,dim
    
class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det
    #--------------------------------------------------------------------------------------------------------------------------------  
if __name__ == '__main__':
    data = pd.read_csv('c172_file_1.csv')
    step_size= 1
    batch = 128
    index_step_length = 31
    epochs = 20
    #---------------------------------------------------------------------------------------------------------------------------------
    labels, X, Y, XX, YY = Splitting_dataset(data, step_size)
    demo = VAE(index_step_length,h_dim=7,z_dim=2)
    demo.double()
    optimizer = torch.optim.RMSprop(demo.parameters(), lr=1e-4)
    
    #print("currParams: " + str(demo.parameters()))
 
    #do i have to evaluate the priors with the vae before I can train the NF model? (at least for actual use)
    
    #check what type of flows are used in OmniAnomaly, since the flows are like an array of models and i dont think that the vae
    #is the thing im supposed to make an array of, since the flow is meant for use in the vae (but could be wrong)
    #demo = NormalizingFlowModel(prior,index_step_length)
    
    #d = torch.distributions.Normal(5,12)
    #print(d.log_prob(torch.Tensor(5)))
    
    idx = 0
    
    anomaly_history = []
    loss_history = []
    avgSum = 0
    avgCount = 0

    for epoch in range(epochs):
        #print(len(XX))
        for i in range(len(XX)):
            localX = torch.tensor(XX[i])
            #print(localX.size())
            recon, mu, logvar = demo(localX)
            #print(recon.size())
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
            plt.plot(loss_history,'g-',label='h 10,z 2')