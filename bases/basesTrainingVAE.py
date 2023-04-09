import os
import torch
from torch import nn
from dataset import TrajectoriesTrainingDataset,TrajectoriesTestingDataset
from torch.utils.data import DataLoader
import numpy as np
import random
from parse_args import parse_yaml,save_yaml
from basesModelVAE import BasesVAE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="trained_bases", help="run id")
cli_args = parser.parse_args()


config_fn = "Panda_Braccio.yaml"
args,config = parse_yaml(config_fn)
# ------------------ USEFUL CONSTANT -----------------
BETA = args[7]
ALPHA = args[8]
DELTA = args[9]
LAMBDA = args[10]
PRIMITIVE_TASK = args[0]
DIM_LATENT = args[1] 
HIDDEN_DIM = args[2]
R1_NAME = args[3]
R2_NAME = args[4]
PATH_R1_TRAj = "trajectories/{}/{}_aligned.txt".format(PRIMITIVE_TASK,R1_NAME)
PATH_R2_TRAj = "trajectories/{}/{}_aligned.txt".format(PRIMITIVE_TASK,R2_NAME)
STATE_DIM_R1 = args[5]
STATE_DIM_R2 = args[6]


num_epochs = args[11]
learning_rate = args[13]
batch_size = args[12]
run_id = cli_args.run_id
print("run id : ",run_id)
print("Training ...")

bases_r1 = BasesVAE(STATE_DIM_R1,DIM_LATENT,HIDDEN_DIM)
bases_r2 = BasesVAE(STATE_DIM_R2,DIM_LATENT,HIDDEN_DIM)

reconstruction_crit = nn.MSELoss()
similarity_crit = nn.MSELoss()

optimizer_r1 = torch.optim.Adam(bases_r1.parameters(), lr=learning_rate)
optimizer_r2 = torch.optim.Adam(bases_r2.parameters(), lr=learning_rate)

train_dataset = TrajectoriesTrainingDataset(PATH_R1_TRAj,PATH_R2_TRAj)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 0)


KL_loss_1 = 0
KL_loss_2 = 0
recon_loss_1 = 0
recon_loss_2 = 0
sim_loss = 0
std_1 = []
std_2 = []
cross_loss_1 = 0
cross_loss_2 = 0
train_loss = 0

for epoch in range(num_epochs):
    for batch_idx, (s1, s2) in enumerate(train_dataloader):
        # ===================forward=====================
        # forward batch of states in encoder/input base
        z_r1,mu_r1,std_r1 = bases_r1.in_(s1)
        z_r2,mu_r2,std_r2 = bases_r2.in_(s2)

        # forward batch of latent state in decoder/output base
        y_r1 = bases_r1.out(z_r1)
        y_r2 = bases_r2.out(z_r2)


        # compute reconstruction loss of source and target
        loss_reconstruction_r1 = ALPHA * reconstruction_crit(y_r1, s1)
        loss_reconstruction_r2 = ALPHA * reconstruction_crit(y_r2, s2)

        loss_KL_r1 = -0.5 * torch.sum(1 + std_r1 - mu_r1.pow(2) - std_r1.exp()) * BETA
        loss_KL_r2 = -0.5 * torch.sum(1 + std_r2 - mu_r2.pow(2) - std_r2.exp()) * BETA

        # Cross reconstruction loss
        y_cross_r1 = bases_r1.out(z_r2)
        y_cross_r2 = bases_r2.out(z_r1)
        loss_cross_r1 = reconstruction_crit(y_cross_r1,s1) * LAMBDA
        loss_cross_r2 = reconstruction_crit(y_cross_r2,s2) * LAMBDA

        loss_similarity = DELTA*(similarity_crit(z_r1,z_r2))

        recon_loss_1 += loss_reconstruction_r1.item()
        recon_loss_2 += loss_reconstruction_r2.item()
        KL_loss_1 += loss_KL_r1.item()
        KL_loss_2 += loss_KL_r2.item()
        sim_loss += loss_similarity.item()
        cross_loss_1 += loss_cross_r1.item()
        cross_loss_2 += loss_cross_r2.item()
        std_1.append(torch.mean(torch.exp(std_r1*0.5).detach(),axis = 0).tolist())
        std_2.append(torch.mean(torch.exp(std_r2*0.5).detach(),axis = 0).tolist())

        total_loss =  loss_similarity + loss_KL_r1 + loss_KL_r2 + loss_cross_r1 + loss_cross_r2 + loss_reconstruction_r1 + loss_reconstruction_r2
        train_loss += total_loss.item()
        # ===================backward====================
        optimizer_r1.zero_grad()
        optimizer_r2.zero_grad()
        total_loss.backward()
        optimizer_r1.step()
        optimizer_r2.step()


    # ===================log========================
    print('epoch [{}/{}], train_loss:{:.10f}'.format(epoch + 1, num_epochs, train_loss/len(train_dataloader)))
    print("sim_loss : ", sim_loss/DELTA/len(train_dataloader))
    print("recon_loss_1 : ", recon_loss_1/ALPHA/len(train_dataloader))
    print("recon_loss_2 : ", recon_loss_2/ALPHA/len(train_dataloader))
    print("KL_loss_1 : ", KL_loss_1/(BETA+1e-8)/len(train_dataloader))
    print("KL_loss_2 : ", KL_loss_2/(BETA+1e-8)/len(train_dataloader))
    print("std1 : ",np.mean(std_1))
    print("std2 : ",np.mean(std_2))
    print("cross_loss_1 : ", cross_loss_1/LAMBDA/len(train_dataloader))
    print("cross_loss_2 : ", cross_loss_2/LAMBDA/len(train_dataloader))
    KL_loss_1 = 0
    KL_loss_2 = 0
    recon_loss_1 = 0
    recon_loss_2 = 0
    sim_loss = 0
    cross_loss_1 = 0
    cross_loss_2 = 0
    std_1 = []
    std_2 = []
    train_loss = 0

    with torch.no_grad():
        test_idx = np.random.randint(0,len(train_dataset))
        (test_item_r1,test_item_r2) = train_dataset.__getitem__(test_idx)

        z_r1,mu_r1,std_r1 = bases_r1.in_(torch.tensor(test_item_r1))
        y_r1 = bases_r1.out(z_r1)

        z_r2,mu_r2,std_r2 = bases_r2.in_(torch.tensor(test_item_r2))
        y_r2 = bases_r2.out(z_r2)

        cross_y_r1 = bases_r1.out(z_r2)
        cross_y_r2 = bases_r2.out(z_r1)
        print('sim [r1/r2], [{}\n/\n{}]'.format(mu_r1.detach().tolist(), mu_r2.detach().tolist()))
        print()
        print()
        print('reconstruction r1 [input/output], [{}\n/\n{}]'.format(test_item_r1.tolist(),y_r1.detach().tolist()))
        print()
        print()
        print('reconstruction r2 [input/output], [{}\n/\n{}]'.format(test_item_r2.tolist(),y_r2.detach().tolist()))
        print()
        print()
        print('cross reconstruction r1 [input/output], [{}\n/\n{}]'.format(test_item_r1.tolist(),cross_y_r1.detach().tolist()))
        print()
        print()
        print('cross reconstruction r2 [input/output], [{}\n/\n{}]'.format(test_item_r2.tolist(),cross_y_r2.detach().tolist()))
        print("-----------------------------------------------------------")
        print()

dir_path = 'models/{}/bases_dim_{}/100k/{}/'.format(PRIMITIVE_TASK,DIM_LATENT,run_id)

# os makdir
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


torch.save(bases_r1.in_.state_dict(), 'models/{}/bases_dim_{}/100k/{}/{}_input.pth'.format(PRIMITIVE_TASK,DIM_LATENT,run_id,R1_NAME,R2_NAME))
torch.save(bases_r1.out.state_dict(), 'models/{}/bases_dim_{}/100k/{}/{}_output.pth'.format(PRIMITIVE_TASK,DIM_LATENT,run_id,R1_NAME,R2_NAME))
torch.save(bases_r2.in_.state_dict(), 'models/{}/bases_dim_{}/100k/{}/{}_input.pth'.format(PRIMITIVE_TASK,DIM_LATENT,run_id,R2_NAME,R1_NAME))
torch.save(bases_r2.out.state_dict(), 'models/{}/bases_dim_{}/100k/{}/{}_output.pth'.format(PRIMITIVE_TASK,DIM_LATENT,run_id,R2_NAME,R1_NAME))

save_yaml(dir_path,run_id,config)


# print total reconstruction error :
x1 = torch.tensor(train_dataset.trajectories_r1)
x2 = torch.tensor(train_dataset.trajectories_r2)
z_r1,mu1,std1 = bases_r1.in_(x1)
z_r2,mu2,std2 = bases_r2.in_(x2)


z_r1 = z_r1.detach().cpu().numpy()
z_r2 = z_r2.detach().cpu().numpy()
mu1 = mu1.detach().cpu().numpy()
mu2 = mu2.detach().cpu().numpy()

print("mean r1 :",np.mean(z_r1,0))
print("mean r2 :",np.mean(z_r2,0))

print("std r1 :",np.std(mu1,0))
print("std r2 :",np.std(mu2,0))


fig = plt.figure(figsize = (8,8))
fig.suptitle('Latent space with PCA')
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('z_1', fontsize = 25)
ax.set_ylabel('z_2', fontsize = 25)
pca = PCA(n_components=2)
X = np.concatenate((z_r1,z_r2,mu1,mu2),axis = 0)
X = StandardScaler().fit_transform(X)
X = pca.fit_transform(X)
Xz1 = X[0:len(train_dataset)]
Xz2 = X[len(train_dataset):2*len(train_dataset)]
Xm1 = X[2*len(train_dataset):3*len(train_dataset)]
Xm2 = X[3*len(train_dataset):4*len(train_dataset)]
colors = ['r', 'g','b','y']
ax.scatter(z_r1[:,0], z_r1[:,1], c = colors[0], s = 20,alpha=0.5)
ax.scatter(z_r2[:,0], z_r2[:,1], c = colors[1], s = 20,alpha=0.5)
ax.scatter(mu1[:,0], mu1[:,1], c = colors[2], s = 20,alpha=0.5) 
ax.scatter(mu2[:,0], mu2[:,1], c = colors[3], s = 20,alpha=0.5) 
targets = ['latent space Panda (sampled)','latent space Braccio (sampled)','latent space Panda (mu)','latent space Braccio (mu)']
ax.legend(targets,loc = 'upper right', fontsize = 25, markerscale = 3.0)
plt.show()