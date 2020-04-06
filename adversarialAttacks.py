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

    original_predictions = torch.stack(original_predictions)
    adversarial_predictions = torch.stack(adversarial_predictions)

    softmax_diff = original_predictions-adversarial_predictions
    softmax_diff_norms = softmax_diff.abs().max(dim=-1)[0]
    return softmax_diff_norms

def softmax_robustness(original_outputs, adversarial_outputs):
    """ This robustness measure is global and it is stricly dependent on the epsilon chosen for the perturbations."""

    softmax_differences = softmax_difference(original_outputs, adversarial_outputs)
    robustness = (torch.ones_like(softmax_differences)-softmax_differences).sum(dim=0)/len(original_outputs)
    # print(softmax_differences)
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


def pgd_attack(model, image, label, epsilon=0.3, alpha=2 / 255, iters=5):

    original_image = copy.deepcopy(image)
    image.requires_grad = True  

    for i in range(iters):
        output = model.forward(image)
        loss = torch.nn.CrossEntropyLoss()(output, label).to(device)
        model.zero_grad()
        loss.backward()

        perturbed_image = image + alpha * image.grad.sign()
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        image = torch.clamp(original_image + eta, min=0, max=1).detach()

    return perturbed_image


def attack(net, x_test, y_test, dataset_name, device, method, filename):

    print(f"\nProducing {method} attacks on {dataset_name}.\n")

    adversarial_attack = []

    for idx in tqdm(range(len(x_test))):
        image = x_test[idx].to(device).unsqueeze(0)
        label = y_test[idx].to(device).argmax(-1).unsqueeze(0)

        if method == "fgsm":
            perturbed_image = fgsm_attack(model=net, image=image, label=label)
        elif method == "pgd":
            perturbed_image = pgd_attack(model=net, image=image, label=label)

        adversarial_attack.append(perturbed_image)

    save_to_pickle(data=adversarial_attack, path=TESTS+filename+"/", 
                   filename=filename+"_"+str(method)+"_attack.pkl")
    return adversarial_attack


def attack_evaluation(model, x_test, x_attack, y_test, device):

    with torch.no_grad():

        original_outputs = []
        adversarial_outputs = []
        original_correct = 0.0
        adversarial_correct = 0.0

        for idx in tqdm(range(len(x_test))):

            image = x_test[idx].to(device).unsqueeze(0)
            attack = x_attack[idx].to(device).unsqueeze(0)
            label = y_test[idx].to(device).argmax(-1).unsqueeze(0)

            original_output = model.forward(image)

            adversarial_output = model.forward(attack)
            original_correct += ((original_output.argmax(-1) == label).sum().item())
            adversarial_correct += ((adversarial_output.argmax(-1) == label).sum().item())

            original_outputs.append(original_output)
            adversarial_outputs.append(adversarial_output)
    
        original_accuracy = 100 * original_correct / len(data_loader.dataset)
        adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)
        
        print(f"\norig_acc = {original_accuracy}\t\tadv_acc = {adversarial_accuracy}", end="\t")
        softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)


def main(args):

    _, _, x_test, y_test, inp_shape, out_size = \
                                        load_dataset(dataset_name=args.dataset, n_inputs=args.inputs)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    dataset, epochs, lr, rel_path = ("mnist", 20, 0.001, TRAINED_MODELS)    
    nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size)
    nn.load(epochs=epochs, lr=lr, device=args.device, rel_path=rel_path)

    x_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=args.dataset, 
                      device=args.device, method="fgsm", filename=nn.filename)
    attack_evaluation(model=nn, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

    bnn = redBNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
                 inference=args.inference, base_net=nn)
    hyperparams = bnn.get_hyperparams(args)
    bnn.load(n_inputs=args.inputs, hyperparams=hyperparams, device=args.device, rel_path=TESTS)
    attack_evaluation(model=bnn, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="adversarial attacks")

    parser.add_argument("--inputs", nargs="?", default=100, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--inference", nargs='?', default="svi", type=str)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--hmc_samples", nargs='?', default=30, type=int)
    parser.add_argument("--warmup", nargs='?', default=10, type=int)
    parser.add_argument("--lr", nargs='?', default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')   

    main(args=parser.parse_args())
