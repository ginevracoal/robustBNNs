import os
import sys
import pyro
import torch
import argparse 
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import decomposition
from pyro.distributions import Normal

from utils import *
from savedir import *
from model_bnn import BNN

### set params

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="fullBNN", type=str, help="fullBNN")
parser.add_argument("--model_idx", default=10, type=int, help="10, 11 (HMC only)")
parser.add_argument("--n_samples", default=50, type=int, help="Number of posterior samples.")
parser.add_argument("--load_model", default=True, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--plot_only", default=True, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--epsilon", default=0.2, type=int, help="Strength of a perturbation.")
parser.add_argument("--same_pca", default=False, type=eval, help="Use same principal components for all subplots.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

# assert pyro.__version__.startswith('1.3.0')

n_inputs_list = [100] if args.debug else [1000, 10000, 60000]

BNN_settings = {"model_10":{"dataset":"mnist", "hidden_size":512, "activation":"leaky", "architecture":"fc2", 
                               "inference":"hmc", "epochs":None, "lr":None, "hmc_samples":args.n_samples, "warmup":100}, 
                "model_11":{"dataset":"fashion_mnist", "hidden_size":512, "activation":"leaky", "architecture":"fc2", 
                           "inference":"hmc", "epochs":None, "lr":None, "hmc_samples":args.n_samples, "warmup":100}}  

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

m = BNN_settings["model_"+str(args.model_idx)]

out_dir = DATA
rel_path = out_dir+'debug/' if args.debug else out_dir

plot_filename = str(m['dataset'])+'_'+str(m['architecture'])+'_'+str(m['inference'])
plot_filename += '_samePCA' if args.same_pca else '_sepPCA'

if args.plot_only:

    _, _, inp_shape, out_size = data_loaders(dataset_name=m['dataset'], n_inputs=1, batch_size=1, shuffle=True)
    net = BNN(m['dataset'], *list(m.values())[1:], inp_shape, out_size)

    df = pd.read_csv(os.path.join(rel_path, net.name, plot_filename+".csv"), index_col=[0])
    print(df)

else:

    df = pd.DataFrame()
    test_loader = data_loaders(dataset_name=m['dataset'], batch_size=128, shuffle=True, n_inputs=60000)[1]

    print("\n=== Train models ===")

    all_weights = []

    for n_inputs in n_inputs_list:

        #### Always using a single chain with batch_size=n_inputs
        train_loader, _, inp_shape, out_size = data_loaders(dataset_name=m['dataset'], n_inputs=n_inputs,
                                                                      batch_size=n_inputs, shuffle=True)

        filename = str(m['dataset'])+'_'+str(m['architecture'])+'_'+str(m['inference'])+'_trainInp='+str(n_inputs)

        net = BNN(m['dataset'], *list(m.values())[1:], inp_shape, out_size)

        if args.load_model:
            net.load(device=args.device, rel_path=out_dir, filename=filename)
        else:
            net.train(train_loader=train_loader, device=args.device, rel_path=out_dir, filename=filename)

        net.evaluate(test_loader=test_loader, device=args.device, n_samples=args.n_samples)

        weights = []
        for sample_idx in range(args.n_samples):
            sampled_net = net.posterior_predictive[sample_idx]

            net_weights = []
            for layer_weights in sampled_net.parameters():
                net_weights.append(layer_weights.flatten())

            net_weights = torch.cat(net_weights)
            weights.append(net_weights)

        all_weights.append(torch.stack(weights))

    print("\n=== PCA fit ===")

    if args.same_pca:
        all_weights = torch.cat(all_weights).detach().cpu().numpy()
        print(f'\nweights.shape = {all_weights.shape}')

        pca = decomposition.PCA(n_components=2)
        pca.fit(all_weights)

    print("\n=== prior samples ===")

    loc = torch.zeros_like(net_weights)
    scale = torch.ones_like(net_weights)
    prior = Normal(loc=loc, scale=scale)

    prior_weights = []
    for _ in range(1000):
        prior_weights.append(prior.rsample())
    prior_weights = torch.stack(prior_weights).detach().cpu().numpy()

    if args.same_pca:
        principal_weights = pca.transform(prior_weights)
    else:
        pca = decomposition.PCA(n_components=2)
        principal_weights = pca.fit_transform(prior_weights)

    for obs in principal_weights:
        df = df.append({'n_samples':1000, 'n_training_points':0, 'x':obs[0], 'y':obs[1]}, ignore_index=True)

    print("\n=== PCA transform posterior samples ===")

    for n_inputs in n_inputs_list:

        filename = str(m['dataset'])+'_'+str(m['architecture'])+'_'+str(m['inference'])+'_trainInp='+str(n_inputs)

        print("\nPCA transform:\n", filename)

        net = BNN(m['dataset'], *list(m.values())[1:], inp_shape, out_size)
        net.load(rel_path=out_dir, device=args.device, filename=filename)

        weights = [] 
        for sample_idx in range(args.n_samples):
            sampled_net = net.posterior_predictive[sample_idx]

            net_weights = []
            for layer_weights in sampled_net.parameters():
                net_weights.append(layer_weights.flatten())

            weights.append(torch.cat(net_weights))

        weights = torch.stack(weights).detach().cpu().numpy() # shape = (n_samples, total_n_weights)

        if args.same_pca:
            principal_weights = pca.transform(weights)

        else:
            pca = decomposition.PCA(n_components=2)
            principal_weights = pca.fit_transform(weights)

        for obs in principal_weights:
            df = df.append({'n_samples':int(args.n_samples), 'n_training_points':n_inputs, 
                            'x':obs[0], 'y':obs[1]}, ignore_index=True)

    print(df.head())
    df.to_csv(os.path.join(rel_path, net.name, plot_filename+".csv"))

######################
# Plot distributions #
######################

from matplotlib.colors import ListedColormap

cmap = plt.get_cmap('rocket', 5)
palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

sns.set_style("darkgrid")
matplotlib.rc('font', **{'size': 9})
fig, ax = plt.subplots(1, len(n_inputs_list)+1, figsize=(10, 3), sharex=False, sharey=False, dpi=150, 
                        facecolor='w', edgecolor='k') 
fig.tight_layout()
fig.subplots_adjust(bottom=0.14)
fig.subplots_adjust(left=0.08)
fig.subplots_adjust(top=0.86)

temp_df = df[df['n_training_points']==0]
sns.kdeplot(data=temp_df, x='x', y='y', ax=ax[0], color=palette[0]) 
ax[0].set_title(f'Prior', weight='bold')

for idx, n_inputs in enumerate(n_inputs_list):
    temp_df = df[df['n_training_points']==n_inputs]
    sns.kdeplot(data=temp_df, x='x', y='y', ax=ax[idx+1], color=palette[idx+1])#, hue='n_training_points')
    ax[idx+1].set_title(f'Posterior\nTraining pts = {n_inputs}', weight='bold')

fig.savefig(os.path.join(TESTS, plot_filename+".png"))
plt.close(fig)