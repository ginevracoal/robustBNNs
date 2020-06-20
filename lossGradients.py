"""
Compute expected loss gradients 
"""

import sys
from savedir import *
import argparse
from tqdm import tqdm
import torch
import copy
from utils import save_to_pickle, load_from_pickle, data_loaders
import numpy as np
import pyro
from model_bnn import BNN, saved_BNNs


DEBUG=False


def loss_gradient(net, image, label, n_samples=None):

    image = image.unsqueeze(0)
    label = label.argmax(-1).unsqueeze(0)

    if n_samples: ## bayesian

        loss_gradients = []

        for i in range(n_samples):
            x_copy = copy.deepcopy(image)
            x_copy.requires_grad = True

            output = net.forward(inputs=x_copy, return_prob=True, n_samples=1, seeds=[i])
            loss = torch.nn.CrossEntropyLoss()(output.to(dtype=torch.double), label)
            net.zero_grad()
            loss.backward()
            loss_gradient = copy.deepcopy(x_copy.grad.data[0])
            loss_gradients.append(loss_gradient)

        loss_gradient = torch.stack(loss_gradients,0).mean(0)

    else: ## non bayesian
        output = net_copy.forward(inputs=x_copy) 

        loss = torch.nn.CrossEntropyLoss()(output.to(dtype=torch.double), label)
        net.zero_grad()
        loss.backward()
        loss_gradient = copy.deepcopy(x_copy.grad.data[0])

    return loss_gradient

# def old_loss_gradient(net, image, label, n_samples=None):

#     image = image.unsqueeze(0)
#     label = label.argmax(-1).unsqueeze(0)

#     x_copy = copy.deepcopy(image)
#     x_copy.requires_grad = True
#     net_copy = copy.deepcopy(net)

#     if n_samples: ## bayesian
#         # output = net_copy.forward(inputs=x_copy, n_samples=n_samples) 

#         output_list = [net_copy.forward(inputs=x_copy, return_logits=True, 
#                        n_samples=1, seeds=[i]) for i in range(n_samples)]
#         output = torch.stack(output_list,0).mean(0)

#     else: ## non bayesian
#         output = net_copy.forward(inputs=x_copy) 

#     loss = torch.nn.CrossEntropyLoss()(output.to(dtype=torch.double), label)
#     net_copy.zero_grad()
    
#     loss.backward()
#     loss_gradient = copy.deepcopy(x_copy.grad.data[0])
#     return loss_gradient

def loss_gradients(net, data_loader, device, filename, savedir, n_samples=None):
    print(f"\n === Loss gradients on {len(data_loader.dataset)} input images:")

    loss_gradients = []
    for images, labels in tqdm(data_loader):
        for i in range(len(images)):
            # pointwise loss gradients
            loss_gradients.append(loss_gradient(net=net, n_samples=n_samples,
                                  image=images[i].to(device), label=labels[i].to(device)))

    loss_gradients = torch.stack(loss_gradients)
    print(f"\nmean = {loss_gradients.mean():.4f} \t var = {loss_gradients.var():.4f}")

    loss_gradients = loss_gradients.cpu().detach().numpy().squeeze()
    save_loss_gradients(loss_gradients, n_samples, filename, savedir)
    return loss_gradients

def save_loss_gradients(loss_gradients, n_samples, filename, savedir):
    save_to_pickle(data=loss_gradients, path=TESTS+savedir, 
                   filename=filename+"_samp="+str(n_samples)+"_lossGrads.pkl")

def load_loss_gradients(n_samples, filename, savedir, relpath=DATA):
    path = relpath+savedir+filename+"_samp="+str(n_samples)+"_lossGrads.pkl"
    return load_from_pickle(path=path)

def compute_vanishing_norms_idxs(loss_gradients, n_samples_list, norm):
    if loss_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should equal the length of `n_samples_list`")

    vanishing_gradients_idxs = []

    print("\nvanishing gradients norms:\n")
    count_van_images = 0
    for image_idx, image_gradients in enumerate(loss_gradients):

        if norm == "linfty":
            gradient_norm = np.max(np.abs(image_gradients[0]))
        elif norm == "l2":
            gradient_norm = np.linalg.norm(image_gradients[0])  
        
        if gradient_norm != 0.0:
            print("image_idx =",image_idx, end="\t")
            count_samples_idx = 0
            for samples_idx, n_samples in enumerate(n_samples_list):

                if norm == "linfty":
                    new_gradient_norm = np.max(np.abs(image_gradients[samples_idx]))
                elif norm == "l2":
                    new_gradient_norm = np.linalg.norm(image_gradients[samples_idx])

                if new_gradient_norm <= gradient_norm:
                    print(new_gradient_norm, end="\t")
                    gradient_norm = copy.deepcopy(new_gradient_norm)
                    count_samples_idx += 1
            if count_samples_idx == len(n_samples_list):
                vanishing_gradients_idxs.append(image_idx)
                print("\tcount=", count_van_images)
                count_van_images += 1
            print("\n")

    print("\nvanishing_gradients_idxs = ", vanishing_gradients_idxs)
    return vanishing_gradients_idxs


def main(args):

    posterior_samples_list=[1,10,100]

    # === load BNN and data ===

    model = saved_BNNs["model_0"]
    dataset, init = list(model.values())[0], list(model.values())[1:]

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name=dataset, batch_size=128, n_inputs=args.inputs, shuffle=False)

    bnn = BNN(dataset, *init, inp_shape, out_size)
    bnn.load(device=args.device, rel_path=DATA)
    filename = bnn.name
    
    # === compute loss gradients ===

    for posterior_samples in posterior_samples_list:
        loss_gradients(net=bnn, n_samples=posterior_samples, savedir=filename+"/", 
                       data_loader=test_loader, device=args.device, filename=filename)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=1000, type=int)
    parser.add_argument("--device", default='cuda', type=str, help='cpu, cuda')   
    main(args=parser.parse_args())