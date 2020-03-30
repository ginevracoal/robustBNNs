from directories import *
import sys
import argparse
from tqdm import tqdm
import torch
import copy
from utils import save_to_pickle, load_from_pickle, data_loaders
import numpy as np
import pyro
from reducedBNN import NN, redBNN


DEBUG=False


def loss_gradient(bnn, n_samples, image, label, device):

    image = image.to(device).unsqueeze(0)
    label = label.to(device).argmax(-1).unsqueeze(0)

    x_copy = copy.deepcopy(image)
    x_copy.requires_grad = True

    bnn_copy = copy.deepcopy(bnn)
    output = bnn_copy.forward(inputs=x_copy, n_samples=n_samples)
    loss = torch.nn.CrossEntropyLoss()(output.to(dtype=torch.double), label)
    bnn_copy.zero_grad()

    loss.backward()
    loss_gradient = copy.deepcopy(x_copy.grad.data[0])
    return loss_gradient


def loss_gradients(bnn, n_samples, data_loader, device, filename):
    print(f"\n === Expected loss gradients on {n_samples} posteriors"
          f" and {len(data_loader.dataset)} input images:")

    loss_gradients = []
    for images, labels in tqdm(data_loader):
        for i in range(len(images)):
            loss_gradients.append(loss_gradient(bnn=bnn, n_samples=n_samples, 
                                      image=images[i], label=labels[i], device=device))

    loss_gradients = torch.stack(loss_gradients)
    print(f"\nexp_mean = {loss_gradients.mean()} \t exp_std = {loss_gradients.std()}")

    loss_gradients = loss_gradients.cpu().detach().numpy().squeeze()
    save_to_pickle(data=loss_gradients, path=TESTS+filename+"/", filename=filename+"_lossGrads.pkl")
    return loss_gradients

def load_loss_gradients(inference, n_inputs, n_samples, filename, relpath=DATA):
    return load_from_pickle(path=relpath+filename+"/"+filename+"_lossGrads.pkl")


def compute_vanishing_norms_idxs(loss_gradients, n_samples_list, norm):
    if loss_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should equal the length of `n_samples_list`")

    vanishing_gradients_idxs = []

    print("\nvanishing gradients norms:")
    count_van_images = 0
    for image_idx, image_gradients in enumerate(loss_gradients):

        if norm == "linfty":
            gradient_norm = np.max(np.abs(image_gradients[0]))
        elif norm == "l2":
            gradient_norm = np.linalg.norm(image_gradients[0])  
        
        if gradient_norm != 0.0:
            print("idx =",image_idx, end="\t")
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
    nn.load(epochs=epochs, lr=lr, rel_path=rel_path, device=args.device)

    # === load reduced BNN ===
    bnn = redBNN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size, 
                 inference=args.inference, base_net=nn)
    hyperparams = bnn.get_hyperparams(args)
    filename = bnn.get_filename(n_inputs=args.inputs, hyperparams=hyperparams)
    bnn.load(n_inputs=args.inputs, hyperparams=hyperparams, rel_path=TESTS, device=args.device)
    
    # === compute loss gradients ===
    n_samples_list = [1,5,10]

    loss_gradients_list = []
    for posterior_samples in n_samples_list:
        loss_gradients_list.append(loss_gradients(bnn=bnn, n_samples=posterior_samples, 
                                   data_loader=test_loader, device=args.device, filename=filename))
    

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