import sys
from directories import *

import argparse
from tqdm import tqdm
import torch
import copy
from utils import save_to_pickle, load_from_pickle, _onehot
import numpy as np
import pyro
from reducedBNN import *


DEBUG=False


def get_filename(inference, n_inputs, n_samples):
    return str(inference)+"_inputs="+str(n_inputs)+"_samples="+str(n_samples)+"_expLossGrads.pkl"

def loss_gradient(bnn, n_samples, image, label, device):

    input_size = image.size(0) * image.size(1) * image.size(2)
    image = image.view(-1, input_size).to(device)
    label = label.to(device).argmax(-1).view(-1)

    x_copy = copy.deepcopy(image)
    bnn_copy = copy.deepcopy(bnn)

    x_copy.requires_grad = True
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

# def categorical_loss_gradients_norms(loss_gradients, n_samples_list, dataset_name, model_idx):
#     loss_gradients = np.array(np.transpose(loss_gradients, (1,0,2)))

#     if loss_gradients.shape[1] != len(n_samples_list):
#         raise ValueError("Second dimension should equal the length of `n_samples_list`")

#     vanishing_idxs = compute_vanishing_grads_idxs(loss_gradients, n_samples_list)
#     const_null_idxs = compute_constantly_null_grads(loss_gradients, n_samples_list)

#     loss_gradients_norms_categories = []
#     for image_idx in range(len(loss_gradients)):
#         if image_idx in vanishing_idxs:
#             loss_gradients_norms_categories.append("vanishing")
#         elif image_idx in const_null_idxs:
#             loss_gradients_norms_categories.append("const_null")
#         else:
#             loss_gradients_norms_categories.append("other")

#     filename = str(dataset_name)+"_bnn_inputs="+str(len(loss_gradients))+\
#                "_samples="+str(n_samples_list)+"_cat_lossGrads_norms"+str(model_idx)+".pkl"
#     save_to_pickle(data=loss_gradients_norms_categories, relative_path=RESULTS, filename=filename)
#     return {"categories":loss_gradients_norms_categories}


def main(args):

    _, test_loader, inp_size, out_size = \
        data_loaders(dataset_name=args.dataset, batch_size=128, n_inputs=args.inputs)

    # load base network
    nn = NN(dataset_name=args.dataset, input_size=inp_size, output_size=out_size, 
            hidden_size=args.hidden_size)
    nn.load(epochs=args.epochs, lr=args.lr)

    # load reduced BNN
    bnn = rBNN(dataset_name=args.dataset, input_size=inp_size, output_size=out_size, 
               hidden_size=args.hidden_size, inference=args.inference, base_net=nn)
    bnn.load(epochs=args.epochs, lr=args.lr)

    # compute expected loss gradients
    filepath = bnn.get_filepath(epochs=args.epochs, lr=args.lr)
    loss_gradients(bnn=bnn, n_samples=args.samples, data_loader=test_loader, 
                            device=args.device, filepath=filepath)



if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="expected loss gradients")

    parser.add_argument("--inputs", nargs="?", default=100, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--lr", nargs='?', default=0.001, type=float)
    parser.add_argument("--hidden_size", nargs='?', default=512, type=int)
    parser.add_argument("--inference", nargs='?', default="svi", type=str)
    parser.add_argument("--samples", nargs='?', default=5, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')   

    main(args=parser.parse_args())