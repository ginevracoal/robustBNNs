import sys
from directories import *

import argparse
from tqdm import tqdm
import torch
import copy
from utils import save_to_pickle, load_from_pickle, data_loaders
import numpy as np
import pyro
from reducedBNN import NN, redBNN
    

DEBUG=False


def get_filename(inference, n_inputs, n_samples):
    return str(inference)+"_inputs="+str(n_inputs)+"_samples="+str(n_samples)+"_expLossGrads.pkl"

def loss_gradient(bnn, n_samples, image, label, device):

    image = image.unsqueeze(0)
    label = label.to(device).argmax(-1).unsqueeze(0)

    x_copy = copy.deepcopy(image)
    x_copy.requires_grad = True

    bnn_copy = copy.deepcopy(bnn)
    output = bnn_copy.forward(inputs=x_copy, n_samples=n_samples, device=device)
    loss = torch.nn.CrossEntropyLoss()(output.to(dtype=torch.double), label)
    bnn_copy.zero_grad()

    loss.backward()
    loss_gradient = copy.deepcopy(x_copy.grad.data[0])
    return loss_gradient


def loss_gradients(bnn, n_samples, data_loader, device, filepath):
    print(f"\n === Expected loss gradients on {n_samples} posteriors"
          f" and {len(data_loader.dataset)} input images:")

    loss_gradients = []
    for images, labels in data_loader:
        for i in tqdm(range(len(images))):
            loss_gradients.append(loss_gradient(bnn=bnn, n_samples=n_samples, 
                                      image=images[i], label=labels[i], device=device))

    loss_gradients = torch.stack(loss_gradients)

    # mean_over_inputs = loss_gradients.mean(0)  
    # std_over_inputs = loss_gradients.std(0)
    # print(f"\nmean_over_inputs[:20] = {mean_over_inputs[:20].cpu().detach()} "
    #       f"\n\nstd_over_inputs[:20] = {std_over_inputs[:20].cpu().detach()}")

    print(f"\nexp_mean = {loss_gradients.mean()} \t exp_std = {loss_gradients.std()}")

    loss_gradients = loss_gradients.cpu().detach().numpy()

    filename = get_filename(inference=bnn.inference, n_inputs=len(data_loader.dataset), 
                            n_samples=n_samples)
    save_to_pickle(data=loss_gradients, path=TESTS+filepath, filename=filename)

    return loss_gradients

def load_loss_gradients(inference, n_inputs, n_samples, filepath, relpath=DATA):
    filename = get_filename(inference, n_inputs, n_samples)
    return load_from_pickle(path=relpath+filepath+filename).squeeze()

# todo: refactor

# def compute_vanishing_grads_idxs(loss_gradients, n_samples_list):
#     if loss_gradients.shape[1] != len(n_samples_list):
#         raise ValueError("Second dimension should equal the length of `n_samples_list`")

#     vanishing_gradients_idxs = []

#     print("\nvanishing gradients norms:")
#     count_van_images = 0
#     for image_idx, image_gradients in enumerate(loss_gradients):
#         # gradient_norm = np.linalg.norm(image_gradients[0])
#         gradient_norm = np.max(np.abs(image_gradients[0]))
#         if gradient_norm != 0.0:
#             print("idx=",image_idx, end="\t\t")
#             count_samples_idx = 0
#             for samples_idx, n_samples in enumerate(n_samples_list):
#                 # new_gradient_norm = np.linalg.norm(image_gradients[samples_idx])
#                 new_gradient_norm = np.max(np.abs(image_gradients[samples_idx]))
#                 if new_gradient_norm <= gradient_norm:
#                     print(new_gradient_norm, end="\t")
#                     gradient_norm = copy.deepcopy(new_gradient_norm)
#                     count_samples_idx += 1
#             if count_samples_idx == len(n_samples_list):
#                 vanishing_gradients_idxs.append(image_idx)
#                 print(", count=", count_van_images)
#                 count_van_images += 1
#             print("\n")

#     print("\nvanishing_gradients_idxs = ", vanishing_gradients_idxs)
#     return vanishing_gradients_idxs

# def compute_constantly_null_grads(loss_gradients, n_samples_list):
#     if loss_gradients.shape[1] != len(n_samples_list):
#         raise ValueError("Second dimension should equal the length of `n_samples_list`")

#     const_null_idxs = []
#     for image_idx, image_gradients in enumerate(loss_gradients):
#         count_samples_idx = 0
#         for samples_idx, n_samples in enumerate(n_samples_list):
#             gradient_norm = np.max(np.abs(image_gradients[samples_idx]))
#             if gradient_norm == 0.0:
#                 count_samples_idx += 1
#         if count_samples_idx == len(n_samples_list):
#             const_null_idxs.append(image_idx)

#     print("\nconst_null_idxs = ", const_null_idxs)
#     return const_null_idxs


def main(args):

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name=args.dataset, batch_size=128, n_inputs=args.inputs, shuffle=True)

    # === load base NN ===
    dataset, epochs, lr, rel_path = ("mnist", 20, 0.001, TRAINED_MODELS)    
    nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size)
    nn.load(epochs=epochs, lr=lr, rel_path=rel_path)

    # === load reduced BNN ===
    bnn = redBNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
                 inference=args.inference, base_net=nn)
    hyperparams = bnn.get_hyperparams(args)
    bnn.load(n_inputs=args.inputs, hyperparams=hyperparams, rel_path=TESTS)
    
    # === compute loss gradients ===
    for posterior_samples in [1,5,10]:
        filepath = bnn.get_filename(n_inputs=args.inputs, hyperparams=hyperparams)+"/"
        loss_gradients(bnn=bnn, n_samples=posterior_samples, data_loader=test_loader, 
                       device=args.device, filepath=filepath)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="expected loss gradients")

    parser.add_argument("--inputs", nargs="?", default=100, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--inference", nargs='?', default="svi", type=str)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--mcmc_samples", nargs='?', default=30, type=int)
    parser.add_argument("--warmup", nargs='?', default=10, type=int)
    parser.add_argument("--lr", nargs='?', default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')   

    main(args=parser.parse_args())