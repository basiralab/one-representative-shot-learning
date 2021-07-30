import argparse
import os
import os.path as osp
import numpy as np
import math
import itertools
import copy
import pickle
from sys import exit

from dgn.model import DGN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable
from torch.distributions import normal, kl

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool
from torch_geometric.utils import get_laplacian, to_dense_adj

import matplotlib.pyplot as plt

from data_utils import MRDataset, create_edge_index_attribute, swap, cross_val_indices, MRDataset2
from model import Generator, Discriminator
from plot import plot, plot_matrix

torch.manual_seed(0)  # To get the same results across experiments
np.random.seed(0)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('running on GPU')
else:
    device = torch.device("cpu")
    print('running on CPU')


# Datasets

h_data = MRDataset2("../data", "lh", subs=989)

# Parameters
batch_size = 1
lr_G = 0.1
lr_D = 0.0002
num_epochs = 300
folds = 3
exp = 1

# Coefficients for loss
i_coeff = 2.0
g_coeff = 2.0
kl_coeff = 0.001

MODEL_PARAMS = {
        "N_ROIs": 35,
        "learning_rate" : 0.0005,
        "n_attr": 2,
        "Linear1" : {"in": 2, "out": 36},
        "conv1": {"in" : 1, "out": 36},
        
        "Linear2" : {"in": 2, "out": 36*24},
        "conv2": {"in" : 36, "out": 24},
        
        "Linear3" : {"in": 2, "out": 24*5},
        "conv3": {"in" : 24, "out": 5} 
    }



def to_1d(vector, n_s, n_r):
    return_array = np.zeros((n_s, int(n_r * (n_r - 1) / 2)))
    
    for k in range(n_s):
        counter = 0
        for i in range(n_r):
            for j in range(i):
                return_array[k][counter] = vector[k][i][j]
                counter += 1
    
    return return_array
    
def to_1d_one_sample(vector, n_r):
    return_array = np.zeros((int(n_r * (n_r - 1) / 2)))

    counter = 0
    for i in range(n_r):
        for j in range(i):
            return_array[counter] = vector[i][j]
            counter += 1
    
    return return_array    

# Training
adversarial_loss = torch.nn.BCELoss().to(device)
identity_loss = torch.nn.L1Loss().to(device)  # Will be used in training
msel = torch.nn.MSELoss().to(device)
mael = torch.nn.L1Loss().to(device)  # Not to be used in training (Measure generator success)

train_ind, val_ind = cross_val_indices(folds, len(h_data))

# Cross Validation
for fold in range(folds):
    train_set, val_set = h_data[list(train_ind[fold])], h_data[list(val_ind[fold])]
    h_data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    h_data_test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    val_step = len(h_data_test_loader)
    
    n_r = train_set[0].x.shape[0]
    n_s = len(train_set)
    few_shot_samples = []
    
    #Store all of the training samples in the fold
    all_samples_x = torch.zeros((n_s, n_r, n_r))
    all_samples_y = torch.zeros((n_s, n_r, n_r))
    all_samples_y2 = torch.zeros((n_s, n_r, n_r))
    for i in range(len(train_set)):
        all_samples_x[i] = train_set[i].x
        all_samples_y[i] = train_set[i].y
        all_samples_y2[i] = train_set[i].y2
        

    #Flatten the samples    
    all_samples_x_flat = to_1d(all_samples_x, n_s, n_r)

    cbt_x = DGN.train_model(
            np.concatenate([all_samples_x.unsqueeze(3), all_samples_x.unsqueeze(3)], axis=3),
            model_params=MODEL_PARAMS,
            n_max_epochs= 100,
            n_folds= 5,
            random_sample_size=10,
            early_stop=True,
            model_name="DGN_test")
            
    cbt_y = DGN.train_model(
            np.concatenate([all_samples_y.unsqueeze(3), all_samples_y.unsqueeze(3)], axis=3),
            model_params=MODEL_PARAMS,
            n_max_epochs= 100,
            n_folds= 5,
            random_sample_size=10,
            early_stop=True,
            model_name="DGN_test")
            
    cbt_y2 = DGN.train_model(
            np.concatenate([all_samples_y2.unsqueeze(3), all_samples_y2.unsqueeze(3)], axis=3),
            model_params=MODEL_PARAMS,
            n_max_epochs= 100,
            n_folds= 5,
            random_sample_size=10,
            early_stop=True,
            model_name="DGN_test")      
            
    cbt_x = (cbt_x[0] + cbt_x[1] + cbt_x[2] + cbt_x[3] + cbt_x[4]) / 5.0
    cbt_y = (cbt_y[0] + cbt_y[1] + cbt_y[2] + cbt_y[3] + cbt_y[4]) / 5.0
    cbt_y2 = (cbt_y2[0] + cbt_y2[1] + cbt_y2[2] + cbt_y2[3] + cbt_y2[4]) / 5.0

    cbt_x = torch.from_numpy(cbt_x)
    cbt_y = torch.from_numpy(cbt_y)
    cbt_y2 = torch.from_numpy(cbt_y2)

    #Create the edge index and attributes from the average
    edge_i, edge_a, _, _ = create_edge_index_attribute(cbt_x)
    edge_i2, edge_a2, _, _ = create_edge_index_attribute(cbt_y)
    edge_i3, edge_a3, _, _ = create_edge_index_attribute(cbt_y2)

    #Create new data from the average values
    temp = Data(x=cbt_x, edge_attr=edge_a, edge_index=edge_i, 
                y=cbt_y, y_edge_attr = edge_a2, y_edge_index = edge_i2, 
                y2=cbt_y2, y2_edge_attr =edge_a3, y2_edge_index = edge_i3).to(device)
                            
    #Add that data to training samples            
    few_shot_samples.append(temp)


    for data in h_data_train_loader:  # Determine the maximum number of samples in a batch
        data_size = data.x.size(0)
        break

    # Create generators and discriminators
    generator = Generator().to(device)
    generator2 = Generator().to(device)
    discriminator = Discriminator().to(device)
    discriminator2 = Discriminator().to(device)

    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_G2 = torch.optim.AdamW(generator2.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_D2 = torch.optim.AdamW(discriminator2.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=0.0)

    total_step = len(h_data_train_loader)
    real_label = torch.ones((data_size, 1)).to(device)
    fake_label = torch.zeros((data_size, 1)).to(device)

    for epoch in range(num_epochs):
        # Reporting
        r, f, d, g, mse_l, mae_l = 0, 0, 0, 0, 0, 0
        r_val, f_val, d_val, g_val, mse_l_val, mae_l_val = 0, 0, 0, 0, 0, 0
        k1_train, k2_train, k1_val, k2_val = 0.0, 0.0, 0.0, 0.0
        r2, f2, d2, g2, mse_l2, mae_l2 = 0, 0, 0, 0, 0, 0
        r_val2, f_val2, d_val2, g_val2, mse_l_val2, mae_l_val2 = 0, 0, 0, 0, 0, 0
        gan1_tr, gan1_val, gan2_tr, gan2_val = 0.0, 0.0, 0.0, 0.0

        # Train
        generator.train()
        discriminator.train()
        generator2.train()
        discriminator2.train()
        for i, data in enumerate(few_shot_samples):
            data = data.to(device)

            optimizer_D.zero_grad()

            # Train the discriminator
            # Create fake data
            fake_y = generator(data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
            fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

            # data: Real source and target
            # fake_data: Real source and generated target
            real_loss = adversarial_loss(discriminator(swapped_data, data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator(fake_data, data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r += real_loss.item()
            f += fake_loss.item()
            d += loss_D.item()
            
            loss_D.backward()
            optimizer_D.step()

            # Train the generator
            optimizer_G.zero_grad()

            # Adversarial Loss
            fake_data.x = generator(data)
            gan_loss = torch.mean(adversarial_loss(discriminator(fake_data, data), real_label[:data.x.size(0), :]))
            gan1_tr += gan_loss.item()

            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data.x.mean(dim=1), fake_data.x.std(dim=1)),
                                       normal.Normal(data.y.mean(dim=1), data.y.std(dim=1))).sum()


            # Identity Loss is included in the end
            loss_G = i_coeff * identity_loss(generator(swapped_data), data.y) + g_coeff * gan_loss + kl_coeff * kl_loss
            g += loss_G.item()
            
            loss_G.backward()
            optimizer_G.step()
            
            k1_train += kl_loss.item()
            mse_l += msel(generator(data), data.y).item()
            mae_l += mael(generator(data), data.y).item()

            # Training of the second part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            optimizer_D2.zero_grad()

            # Train the discriminator2

            # Create fake data for t2 from fake data for t1
            fake_data.x = fake_data.x.detach()
            fake_y2 = generator2(fake_data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
            fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

            # fake_data: Data generated for t1
            # fake_data2: Data generated for t2 using generated data for t1
            # swapped_data2: Real t2 data
            real_loss = adversarial_loss(discriminator2(swapped_data2, fake_data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator2(fake_data2, fake_data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r2 += real_loss.item()
            f2 += fake_loss.item()
            d2 += loss_D.item()

            loss_D.backward()
            optimizer_D2.step()

            # Train generator2
            optimizer_G2.zero_grad()

            # Adversarial Loss
            fake_data2.x = generator2(fake_data)
            gan_loss = torch.mean(adversarial_loss(discriminator2(fake_data2, fake_data), real_label[:data.x.size(0), :]))
            gan2_tr += gan_loss.item()


            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data2.x.mean(dim=1), fake_data2.x.std(dim=1)),
                                       normal.Normal(data.y2.mean(dim=1), data.y2.std(dim=1))).sum()

            # Identity Loss
            loss_G = i_coeff * identity_loss(generator(swapped_data2), data.y2) + g_coeff * gan_loss + kl_coeff * kl_loss
            g2 += loss_G.item()

            loss_G.backward()
            optimizer_G2.step()

            k2_train += kl_loss.item()
            mse_l2 += msel(generator2(fake_data), data.y2).item()
            mae_l2 += mael(generator2(fake_data), data.y2).item()

        if((epoch + 1) % 25 != 0):
            continue
        
        # Validate
        generator.eval()
        discriminator.eval()
        generator2.eval()
        discriminator2.eval()

        for i, data in enumerate(h_data_test_loader):
            data = data.to(device)
            # Train the discriminator
            # Create fake data
            fake_y = generator(data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
            fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

            # data: Real source and target
            # fake_data: Real source and generated target
            real_loss = adversarial_loss(discriminator(swapped_data, data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator(fake_data, data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r_val += real_loss.item()
            f_val += fake_loss.item()
            d_val += loss_D.item()

            # Adversarial Loss
            fake_data.x = generator(data)
            gan_loss = torch.mean(adversarial_loss(discriminator(fake_data, data), real_label[:data.x.size(0), :]))
            gan1_val += gan_loss.item()


            kl_loss = kl.kl_divergence(normal.Normal(fake_data.x.mean(dim=1), fake_data.x.std(dim=1)),
                                       normal.Normal(data.y.mean(dim=1), data.y.std(dim=1))).sum()

            # Identity Loss

            loss_G = i_coeff * identity_loss(generator(swapped_data), data.y) + g_coeff * gan_loss * kl_coeff * kl_loss
            g_val += loss_G.item()
            mse_l_val += msel(generator(data), data.y).item()
            mae_l_val += mael(generator(data), data.y).item()
            k1_val += kl_loss.item()

            # Second GAN

            # Create fake data for t2 from fake data for t1
            fake_data.x = fake_data.x.detach()
            fake_y2 = generator2(fake_data)
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
            fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

            # fake_data: Data generated for t1
            # fake_data2: Data generated for t2 using generated data for t1
            # swapped_data2: Real t2 data
            real_loss = adversarial_loss(discriminator2(swapped_data2, fake_data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator2(fake_data2, fake_data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r_val2 += real_loss.item()
            f_val2 += fake_loss.item()
            d_val2 += loss_D.item()

            # Adversarial Loss
            fake_data2.x = generator2(fake_data)
            gan_loss = torch.mean(adversarial_loss(discriminator2(fake_data2, fake_data), real_label[:data.x.size(0), :]))
            gan2_val += gan_loss.item()


            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data2.x.mean(dim=1), fake_data2.x.std(dim=1)),
                                       normal.Normal(data.y2.mean(dim=1), data.y2.std(dim=1))).sum()
            k2_val += kl_loss.item()

            # Identity Loss
            loss_G = i_coeff * identity_loss(generator(swapped_data2), data.y2) + g_coeff * gan_loss + kl_coeff * kl_loss
            g_val2 += loss_G.item()
            mse_l_val2 += msel(generator2(fake_data), data.y2).item()
            mae_l_val2 += mael(generator2(fake_data), data.y2).item()




        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'[Train]: D Loss: {d / total_step:.5f}, G Loss: {g / total_step:.5f} R Loss: {r / total_step:.5f}, F Loss: {f / total_step:.5f}, MSE: {mse_l / total_step:.5f}, MAE: {mae_l / total_step:.5f}')
        print(f'[Val]: D Loss: {d_val / val_step:.5f}, G Loss: {g_val / val_step:.5f} R Loss: {r_val / val_step:.5f}, F Loss: {f_val / val_step:.5f}, MSE: {mse_l_val / val_step:.5f}, MAE: {mae_l_val / val_step:.5f}')
        print(f'[Train]: D2 Loss: {d2 / total_step:.5f}, G2 Loss: {g2 / total_step:.5f} R2 Loss: {r2 / total_step:.5f}, F2 Loss: {f2 / total_step:.5f}, MSE: {mse_l2 / total_step:.5f}, MAE: {mae_l2 / total_step:.5f}')
        print(f'[Val]: D2 Loss: {d_val2 / val_step:.5f}, G2 Loss: {g_val2 / val_step:.5f} R2 Loss: {r_val2 / val_step:.5f}, F2 Loss: {f_val2 / val_step:.5f}, MSE: {mse_l_val2 / val_step:.5f}, MAE: {mae_l_val2 / val_step:.5f}')


	# Save the models
    torch.save(generator.state_dict(), "../weights/generator_" + str(fold) + "_" + str(epoch) + "_" + str(exp))
    torch.save(discriminator.state_dict(), "../weights/discriminator_" + str(fold) + "_" + str(epoch) + "_" + str(exp))
    torch.save(generator2.state_dict(),
			   "../weights/generator2_" + str(fold) + "_" + str(epoch) + "_" + str(exp))
    torch.save(discriminator2.state_dict(),
			   "../weights/discriminator2_" + str(fold) + "_" + str(epoch) + "_" + str(exp))

    del generator
    del discriminator

    del generator2
    del discriminator2

