import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def differentiable_one_hot_converter(y):
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

def conv_block(in_channels, out_channels, func = nn.ReLU(),batch_norm = 1, max_pool = 1, padding = 1):
    '''
    returns a block conv-bn-relu-pool
    '''
    conv = torch.nn.Sequential()
    if padding == 1:
        conv.add_module("conv",nn.Conv2d(in_channels, out_channels, 3, padding=1))
    else:
        conv.add_module("conv",nn.Conv2d(in_channels, out_channels, 3, padding=0))
    if batch_norm == 1:
        conv.add_module("batch_norm",nn.BatchNorm2d(out_channels))

    if func is not None:
        conv.add_module("activation_function",func)

    if max_pool == 1:
        conv.add_module("max_pool",nn.MaxPool2d(2))

    return conv


class AttPredictor(nn.Module):
    def __init__(self, x_dim=3, hid_dim_list=[64,128,256], att_size=312):
        super(AttPredictor, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim_list[0],nn.ReLU()),
            conv_block(hid_dim_list[0], hid_dim_list[1]),
            conv_block(hid_dim_list[1], hid_dim_list[2]),
            conv_block(hid_dim_list[2], att_size),
        )
        self.linear = nn.Linear(att_size,att_size)

    def forward(self, x):
        x_features_reshape = self.encoder(x) #(n_way*(n_query+n_support),h,w,c)
        x_features_flatten = torch.flatten(torch._adaptive_avg_pool2d(x_features_reshape,(1,1)), start_dim = 1)
        return (torch.tanh(self.linear(x_features_flatten))+1)/2

class AttSelector(nn.Module):
    def __init__(self, tau = 0.5, att_size = 312):
        super(AttSelector, self).__init__()
        
        self.lstm = nn.LSTM(att_size,100,num_layers=1,bidirectional = True,batch_first=True)
        self.fn_inp_size = 200
        self.fn = nn.Sequential(nn.Linear(self.fn_inp_size,att_size),)
        self.sigmoid = nn.Sigmoid()
        
        self.tau = tau
        self.att_size = att_size
        self.gumbel = torch.distributions.gumbel.Gumbel(0,1)
        self.softmax = nn.Softmax(dim=1)
                                              
    def forward(self, x): 
        h_n = self.lstm(torch.unsqueeze(x,0))[1][0]  

        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        probs = self.sigmoid(self.fn(h_n))
        if not self.training:
            return (probs>0.5).float().to(device),probs

        probs_copy = probs.clone()
        
        probs_copy[probs==0] = probs[probs==0] + 0.00001 # This value is added to prevent zero in log
        probs_copy[probs==1] = probs[probs==1] - 0.00001

        log_probs = torch.log(probs_copy) + torch.tensor(np.random.gumbel(0, 1, self.att_size)).to(device)
        
        log_probs_minus = torch.log(1-probs_copy) + torch.tensor(np.random.gumbel(0, 1, self.att_size)).to(device)
        log_probs = log_probs.contiguous().view(-1,1)
        log_probs_minus = log_probs_minus.contiguous().view(-1,1)
        logs = torch.cat([log_probs,log_probs_minus],dim=1)
        logs = logs/self.tau
        soft = self.softmax(logs)
        soft_hard = differentiable_one_hot_converter(soft)
        return soft_hard[:,0], probs_copy

class AttSelectorUnknown(nn.Module):
    def __init__(self, tau, att_size, index_size):
        super(AttSelectorUnknown, self).__init__()
        
        self.lstm = nn.LSTM(att_size,100,num_layers=1,bidirectional = True,batch_first=True)
        self.fn_inp_size = 200
        self.fn = nn.Sequential(nn.Linear(self.fn_inp_size,index_size),)
        self.sigmoid = nn.Sigmoid()
        
        self.tau = tau
        self.att_size = att_size
        self.index_size = index_size
        self.gumbel = torch.distributions.gumbel.Gumbel(0,1)
        self.softmax = nn.Softmax(dim=1)
                                              
    def forward(self, x): #(n_way*n_support,312)
        h_n = self.lstm(torch.unsqueeze(x,0))[1][0]  #The output of lstm is  output, (h_n, c_n)
        #h_n size : 2*num_batches*hidden_size    here num_batches = 1

        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        probs = self.sigmoid(self.fn(h_n))
        if not self.training:
            return (probs>0.5).float().to(device),probs

        probs_copy = probs.clone()
        
        probs_copy[probs==0] = probs[probs==0] + 0.00001
        probs_copy[probs==1] = probs[probs==1] - 0.00001

        log_probs = torch.log(probs_copy) + torch.tensor(np.random.gumbel(0, 1, self.index_size)).to(device)
        
        log_probs_minus = torch.log(1-probs_copy) + torch.tensor(np.random.gumbel(0, 1, self.index_size)).to(device)
        log_probs = log_probs.contiguous().view(-1,1)
        log_probs_minus = log_probs_minus.contiguous().view(-1,1)
        logs = torch.cat([log_probs,log_probs_minus],dim=1)
        logs = logs/self.tau
        soft = self.softmax(logs)
        soft_hard = differentiable_one_hot_converter(soft)
        return soft_hard[:,0], probs_copy

class MutualHelper(nn.Module):
    def __init__(self, att_size_overall):
        super(MutualHelper, self).__init__()
        
        self.fn_inp_size = att_size_overall
        self.net = nn.Sequential(nn.Linear(self.fn_inp_size,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,1),
        )                      
    def forward(self, x):
        return self.net(x)