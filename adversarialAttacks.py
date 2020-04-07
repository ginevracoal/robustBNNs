import sys
from directories import *
import argparse
from tqdm import tqdm
import pyro
import random
import copy
import torch
from reducedBNN import NN, redBNN
from utils import load_dataset, save_to_pickle, load_from_pickle
import numpy as np
from torch.utils.data import DataLoader


DEBUG=False


#######################
# robustness measures #
#######################

def softmax_difference(original_predictions, adversarial_predictions):
    """
    Compute the expected l-inf norm of the difference between predictions and adversarial predictions.
    This is point-wise robustness measure.
    """
    if len(original_predictions) != len(adversarial_predictions):
        raise ValueError("\nInput arrays should have the same length.")

    softmax_diff = original_predictions-adversarial_predictions
    softmax_diff_norms = softmax_diff.abs().max(dim=-1)[0]
    return softmax_diff_norms

def softmax_robustness(original_outputs, adversarial_outputs):
    """ This robustness measure is global and it is stricly dependent on the epsilon chosen for the 
    perturbations."""

    softmax_differences = softmax_difference(original_outputs, adversarial_outputs)
    robustness = (torch.ones_like(softmax_differences)-softmax_differences).sum(dim=0)/len(original_outputs)
    print(f"softmax_robustness = {robustness.item():.2f}")
    return robustness.item()


#######################
# adversarial attacks #
#######################

def fgsm_attack(model, image, label, epsilon=0.3):

    image.requires_grad = True

    output = model.forward(image)
    loss = torch.nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()
    image_grad = image.grad.data

    perturbed_image = image + epsilon * image_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def pgd_attack(model, image, label, epsilon=0.3, alpha=2/255, iters=40):

    original_image = copy.deepcopy(image)

    for i in range(iters):
        image.requires_grad = True  
        output = model.forward(image)
        loss = torch.nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()

        perturbed_image = image + alpha * image.grad.data.sign()
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        image = torch.clamp(original_image + eta, min=0, max=1).detach()

    perturbed_image = image
    return perturbed_image


def attack(net, x_test, y_test, dataset_name, device, method, filename):

    print(f"\nProducing {method} attacks on {dataset_name}:")

    adversarial_attack = []

    for idx in tqdm(range(len(x_test))):
        image = x_test[idx].unsqueeze(0).to(device)
        label = y_test[idx].argmax(-1).unsqueeze(0).to(device)

        if method == "fgsm":
            perturbed_image = fgsm_attack(model=net, image=image, label=label)
        elif method == "pgd":
            perturbed_image = pgd_attack(model=net, image=image, label=label)

        adversarial_attack.append(perturbed_image)

    # concatenate list of tensors 
    adversarial_attack = torch.cat(adversarial_attack)

    save_to_pickle(data=adversarial_attack, path=TESTS+filename+"/", 
                   filename=filename+"_"+str(method)+"_attack.pkl")
    return adversarial_attack

def load_attack(model, method, rel_path=TESTS):
    path = rel_path+model.filename+"/"+model.filename+"_"+str(method)+"_attack.pkl"
    return load_from_pickle(path=path)

def attack_evaluation(model, x_test, x_attack, y_test, device):
    print(f"\nEvaluating against the attacks:")

    model.to(device)
    if hasattr(model, 'net'):
        model.net.to(device) # fixed layers in BNN

    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=128, shuffle=False)
    attack_loader = DataLoader(dataset=list(zip(x_attack, y_test)), batch_size=128, shuffle=False)

    with torch.no_grad():

        original_outputs = []
        original_correct = 0.0
        for images, labels in test_loader:
            out = model.forward(images)
            original_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
            original_outputs.append(out)

        adversarial_outputs = []
        adversarial_correct = 0.0
        for attacks, labels in attack_loader:
            out = model.forward(attacks)
            adversarial_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
            adversarial_outputs.append(out)

        original_accuracy = 100 * original_correct / len(x_test)
        adversarial_accuracy = 100 * adversarial_correct / len(x_test)
        print(f"\ntest accuracy = {original_accuracy}\tadversarial accuracy = {adversarial_accuracy}",
              end="\t")

        original_outputs = torch.cat(original_outputs)
        adversarial_outputs = torch.cat(adversarial_outputs)
        softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)


########
# main #
########

def main(args):

    _, _, x_test, y_test, inp_shape, out_size = \
                                        load_dataset(dataset_name=args.dataset, n_inputs=args.inputs)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    dataset, epochs, lr, rel_path = ("mnist", 20, 0.001, DATA)    
    nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size)
    nn.load(epochs=epochs, lr=lr, device=args.device, rel_path=rel_path)

    x_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=args.dataset, 
                      device=args.device, method=args.attack, filename=nn.filename)
    # x_attack = load_attack(model=nn, method=args.attack, rel_path=DATA)

    attack_evaluation(model=nn, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

    bnn = redBNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
                 inference=args.inference, base_net=nn)
    hyperparams = bnn.get_hyperparams(args)
    bnn.load(n_inputs=args.inputs, hyperparams=hyperparams, device=args.device, rel_path=TESTS)
    attack_evaluation(model=bnn, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="adversarial attacks")

    parser.add_argument("--inputs", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist, cifar, fashion_mnist")
    parser.add_argument("--attack", default="fgsm", type=str, help="fgsm, pgd")
    parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--hmc_samples", default=30, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")   

    main(args=parser.parse_args())
