"""
Ensemble Neural Network model.
"""

from savedir import *
from utils import *
import os
import argparse
import math 
import numpy as np
from model_nn import NN, saved_NNs


class Ensemble_NN(NN):

    def __init__(self, dataset_name, hidden_size, activation, architecture, 
                 epochs, lr, input_shape, output_size, ensemble_size):
        super(Ensemble_NN, self).__init__(dataset_name, input_shape, output_size,
            hidden_size, activation, architecture, lr, epochs)
        self.ensemble_size = ensemble_size
        self.input_shape = input_shape
        self.random_seeds = range(0,ensemble_size)
        self.name = self.get_name(ensemble_size)
        self.ensemble_models = {}

    def get_name(self, ensemble_size, *args, **kwargs):
        
        name = str(self.dataset_name)+"_ensemble_hid="+\
               str(self.hidden_size)+"_act="+str(self.activation)+\
               "_arch="+str(self.architecture)+"_size="+str(ensemble_size)
        return name

    def save(self, seed=None, *args, **kwargs):

        savedir = self.name+"/weights"

        if seed:
            self.ensemble_models[str(seed)].save(savedir=savedir, seed=seed)

        else:
            for idx, net in self.ensemble_models.items():
                net.save(savedir=savedir, seed=idx)

    def load(self, device, rel_path=TESTS):
        
        savedir = self.name+"/weights"
        for seed in self.random_seeds:

            net = NN(dataset_name=self.dataset_name, input_shape=self.input_shape, 
                output_size=self.output_size, hidden_size=self.hidden_size, 
                activation=self.activation, architecture=self.architecture, 
                epochs=self.epochs, lr=self.lr)

            net.load(device=device, savedir=savedir, seed=seed, rel_path=rel_path)
            self.ensemble_models[str(seed)]=net

    def forward(self, inputs, n_samples, *args, **kwargs):

        if n_samples is not None:
            if n_samples > self.ensemble_size:
                raise ValueError("Maximum number of samples allowed is ", self.ensemble_size)

        selected_models = list(self.ensemble_models.values())[:n_samples]
        ensemble_pred = [net.model(inputs) for net in selected_models]
        ensemble_pred = torch.stack(ensemble_pred, 0)
        pred = ensemble_pred.mean(0)
        return pred

    def train(self, x_train, y_train, device):

        for seed in self.random_seeds:

            batch_size = 100 #random.choice([32,64,128,256])
            train_loader = DataLoader(dataset=list(zip(x_train, y_train)), 
                                    batch_size=batch_size, shuffle=True)   

            net = NN(dataset_name=self.dataset_name, input_shape=self.input_shape, 
                output_size=self.output_size, hidden_size=self.hidden_size, 
                activation=self.activation, architecture=self.architecture, 
                epochs=self.epochs, lr=self.lr)
            net.train(train_loader=train_loader, device=device, seed=seed, save=False)
            self.ensemble_models[str(seed)]=net
            self.save(seed=seed)

    def evaluate(self, test_loader, device, n_samples, *args, **kwargs):

        if n_samples > self.ensemble_size:
            raise ValueError("Maximum number of samples allowed is ", self.ensemble_size)
       
        self.to(device)

        with torch.no_grad():

            correct_predictions = 0.0

            for x_batch, y_batch in test_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).argmax(-1)
                outputs = self(x_batch, n_samples=n_samples)
                predictions = outputs.argmax(-1)
                correct_predictions += (predictions == y_batch).sum()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("\nAccuracy: %.2f%%" % (accuracy))
            return accuracy


def main(args):
    
    rel_path=DATA if args.savedir=="DATA" else TESTS

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    dataset, hid, activ, arch, ep, lr = saved_NNs["model_"+str(args.model_idx)].values()
    x_train, y_train, x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=dataset, n_inputs=args.n_inputs, shuffle=True)

    net = Ensemble_NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size, 
            hidden_size=hid, activation=activ, architecture=arch, epochs=ep, lr=lr, 
            ensemble_size = args.ensemble_size)
   
    if args.train:
        net.train(x_train=x_train[:args.n_inputs], y_train=y_train[:args.n_inputs], 
                    device=args.device)
    else:
        net.load(device=args.device, rel_path=rel_path)

    if args.test:
        test_loader = DataLoader(dataset=list(zip(x_test[:args.n_inputs], y_test[:args.n_inputs])), batch_size=64, shuffle=True)   
        net.evaluate(test_loader=test_loader, device=args.device, n_samples=args.ensemble_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=60000, type=int, help="number of input points")
    parser.add_argument("--model_idx", default=0, type=int, help="choose idx from saved_BNNs")
    parser.add_argument("--ensemble_size", default=100, type=int)
    parser.add_argument("--train", default=True, type=eval, help="train or load saved model")
    parser.add_argument("--test", default=True, type=eval, help="evaluate on test data")
    parser.add_argument("--savedir", default='DATA', type=str, help="choose dir for loading the NN: DDATA, TESTS")  
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
    main(args=parser.parse_args())
