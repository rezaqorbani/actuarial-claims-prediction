import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler




def poisson_deviance_loss(pred, target):
        
    # weight each sample by its percentage of the total exposure for the entire dataset
    
    z = 2 * torch.empty_like(pred).copy_(pred)
    ind = (target[:, 0:1] > 0).nonzero()

    z[ind[:, 0]] = 2 * torch.mul(target[ind[:, 0], 0:1], torch.log(torch.div(target[ind[:, 0], 0:1], pred[ind[:, 0]]))) -  2* target[ind[:, 0], 0:1] + 2* pred[ind[:, 0]]

    return torch.sum(torch.mul(z, target[:, 2:3]))


def loss_batch(model, loss_func, xb, yb, opt=None):

    loss = loss_func(model(xb), yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(xb)


def fit(epochs, model, train_loss_func, test_loss_func, opt, scheduler, train_dl, valid_dl):
    
    for epoch in range(epochs):
        
        model.train()        
        
        for xb, yb in train_dl:
            loss_batch(model, train_loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, test_loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(losses)

        scheduler.step()

        print(f'Epoch: {epoch}, Validation loss: {val_loss}')


def sampling_weights(targets, exposure, ratio):

    ## partition data into two sets one with no claim and the other with >=1 claim
    ## after normalizing with frequency to give equal weight to the two classes then use "ratio" to decide on the final balance
    
    n = targets.shape[0]
    ind = (targets == 0).nonzero()
    n0 = ind[0].shape[0]

    ww = [1 / n0,  1 / (n - n0)]
    ww[0] = ratio[0]  * ww[0]
    ww[1] = ratio[1]  * ww[1]
    
    weights = [ww[1] if t > 0 else ww[0] for t in targets]

    return weights


class myNet(nn.Module):
    
    def __init__(self, dim_emb, cat_ind, num_cats, cts_ind, n_hidden):
        super(myNet, self).__init__()

        self.cat_ind = cat_ind
        self.cts_ind = cts_ind
        
        n = len(cat_ind)
        n_cts = len(cts_ind)
        
        self.embedding = []
        for ii in num_cats:
            self.embedding.append(torch.nn.Embedding(int(ii), dim_emb))

        
        self.cts_embedding = []
        for i in range(n_cts):
            self.cts_embedding.append(nn.Linear(1, dim_emb))
        

        self.linear1 = nn.Linear((n+n_cts)*dim_emb, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)



    def forward(self, x):

        ## perform the embedding of the categorical variables
        out = self.embedding[0](x[:, self.cat_ind[0]].int())

        for i in range(1, len(self.cat_ind)):
            ii = self.cat_ind[i]
            e = self.embedding[i](x[:, ii].int())
            e = F.relu(e)            
            out = torch.cat((out,e), 1)

        ## perform the embedding of the non-categorical variables
        for i in range(len(self.cts_ind)):
            ii = self.cts_ind[i]
            e = self.cts_embedding[i](x[:, ii:ii+1])
            e = F.relu(e)
            out = torch.cat((out,e), 1)

        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.exp(out)  # needed to ensure a positive output
        
        return out


def  train_network(emb_params, arch_params, train_params, train_data, val_data):

    ## set seed for debugging purposes    
    torch.manual_seed(100)

    ## initialize the network
    my_nn = myNet(emb_params['dim'], emb_params['cat_ind'], emb_params['num_cats'], emb_params['dis_ind'],
                  arch_params['n_hidden'])
        
    ############### set up the dataloaders ##################
    
    loss_train_weights = train_data['outputs'][:, 1:] / np.sum(train_data['outputs'][:, 1:])
    loss_val_weights = val_data['outputs'][:, 1:] / np.sum(val_data['outputs'][:, 1:])
    
    train_data['outputs'] = np.c_[train_data['outputs'], loss_train_weights]
    val_data['outputs'] = np.c_[val_data['outputs'], loss_val_weights]        

    ## convert arrays to torch tensors + cast to float32
    train_inputs = torch.from_numpy(train_data['inputs']).type(torch.float32)
    train_outputs = torch.from_numpy(train_data['outputs']).type(torch.float32)
    
    val_inputs = torch.from_numpy(val_data['inputs']).type(torch.float32)
    val_outputs = torch.from_numpy(val_data['outputs']).type(torch.float32)

    ## set up the TensorDataset
    train_ds = TensorDataset(train_inputs, train_outputs)
    val_ds = TensorDataset(val_inputs, val_outputs)

    if train_params['use_balanced_sampling']:
        n_train = len(train_outputs)
        samp_weights = sampling_weights(train_data['outputs'][:, 0:1], train_data['outputs'][:, 1:],
                                        train_params['balance ratio'])

        sampler = WeightedRandomSampler(samp_weights, n_train, replacement=True)
        train_dl = DataLoader(train_ds, batch_size=train_params['nb'], sampler = sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=train_params['nb'], shuffle=True)
        
    val_dl = DataLoader(val_ds, batch_size=2*train_params['nb'])
    ###############################################################    
    
    ###### set up the optimizer ##########################
    
    if train_params['optimizer'] == 'sgd':        
        opt = optim.SGD(my_nn.parameters(), lr= train_params['lr'], momentum=0.8, weight_decay=train_params['weight_decay'])
        
    elif train_params['optimizer'] == 'adamw':        
        opt = optim.AdamW(my_nn.parameters(), lr = train_params['lr'], weight_decay=train_params['weight_decay'])
    ###############################################################            


    ###### set up scheduler for the learning rate ######
    # scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=4)
    #scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    #scheduler = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, train_params['n_epochs'], eta_min=1e-4)    


    ###### do the training ######
    fit(train_params['n_epochs'], my_nn, poisson_deviance_loss, poisson_deviance_loss, opt, scheduler, train_dl, val_dl)

    ###### make predictions on the validation set ######
    val_fpred = my_nn(val_inputs);

    return val_fpred.detach().numpy(), my_nn


def make_predictions(model, inputs):

    torch_inputs = torch.from_numpy(inputs).type(torch.float32)
    f_pred = model(torch_inputs)

    return f_pred.detach().numpy()
    
