import argparse
import os
from directories import *

import pyro
import torch
from torch import nn
import torch.nn.functional as nnf
import numpy as np
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
import torch.optim as torchopt
from pyro import poutine
import pyro.optim as pyroopt
import torch.nn.functional as F
from pyro.infer.mcmc import MCMC, HMC
from pyro.infer.mcmc.util import predictive
from pyro.infer.abstract_infer import TracePredictive
from pyro.distributions import OneHotCategorical, Normal, Categorical

from utils import *
# from lossGradients import expected_loss_gradients


softplus = torch.nn.Softplus()


DEBUG=False


class NN(nn.Module):

	def __init__(self, dataset_name, input_shape, output_size):
		super(NN, self).__init__()
		self.dataset_name = dataset_name
		in_channels = input_shape[0]

		if self.dataset_name == "mnist":
			self.l1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5),
									nn.LeakyReLU(),
									nn.MaxPool2d(kernel_size=2))
			self.l2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5),
									nn.LeakyReLU(),
									nn.MaxPool2d(kernel_size=2))
			self.out = nn.Linear(4*4*64, output_size)
		else:
			raise NotImplementedError()

	def get_filename(self, dataset_name, epochs, lr):
		return str(dataset_name)+"_nn_ep="+str(epochs)+"_lr="+str(lr)

	def forward(self, inputs):
		x = inputs

		if self.dataset_name == "mnist":
			x = self.l1(x)
			x = self.l2(x)
			x = x.view(x.size(0), -1)
			x = self.out(x)
		else:
			raise NotImplementedError()

		output = nn.Softmax(dim=-1)(x)
		return output

	def save(self, epochs, lr):
		filename = self.get_filename(self.dataset_name, epochs, lr)+"_weights.pt"
		os.makedirs(os.path.dirname(TESTS), exist_ok=True)
		print("\nSaving: ", TESTS+filename)
		torch.save(self.state_dict(), TESTS+filename)

		if DEBUG:
			print("\nCheck saved weights:")
			print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
			print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

	def load(self, epochs, lr, rel_path=TESTS):
		filename = self.get_filename(self.dataset_name, epochs, lr)+"_weights.pt"
		print("\nLoading: ", rel_path+filename)
		self.load_state_dict(torch.load(rel_path+filename))
		print("\n", list(self.state_dict().keys()), "\n")

		if DEBUG:
			print("\nCheck loaded weights:")	
			print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
			print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

	def train(self, train_loader, epochs, lr, device="cpu"):
		print("\n == NN training ==")
		random.seed(0)

		criterion = nn.CrossEntropyLoss()
		optimizer = torchopt.Adam(params=self.parameters(), lr=0.001)

		start = time.time()
		for epoch in range(epochs):
			total_loss = 0.0
			correct_predictions = 0.0
			accuracy = 0.0
			total = 0.0
			n_inputs = 0

			for images, labels in train_loader:
				n_inputs += len(images)
				images = images.to(device)
				labels = labels.to(device).argmax(-1)
				total += labels.size(0)

				optimizer.zero_grad()
				outputs = self.forward(images)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				predictions = outputs.argmax(dim=1)
				total_loss += loss.data.item() / len(train_loader.dataset)
				correct_predictions += (predictions == labels).sum()
				accuracy = 100 * correct_predictions / len(train_loader.dataset)

			# print(self.l2.0.weight.data)

			print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
				  end="\t")

		execution_time(start=start, end=time.time())
		self.save(epochs=epochs, lr=lr)

	def evaluate(self, test_loader, device):

		with torch.no_grad():

			correct_predictions = 0.0

			for images, labels in test_loader:

				images = images.to(device)
				labels = labels.to(device).argmax(-1)
				outputs = self(images)
				predictions = outputs.argmax(dim=1)
				correct_predictions += (predictions == labels).sum()

			accuracy = 100 * correct_predictions / len(test_loader.dataset)
			print("\nAccuracy: %.2f%%" % (accuracy))
			return accuracy


class rBNN(nn.Module):

	def __init__(self, dataset_name, input_shape, output_size, inference, base_net):
		super(rBNN, self).__init__()
		self.dataset_name = dataset_name
		self.inference = inference
		self.net = base_net

	def get_filepath(self, epochs, lr):
		return str(self.dataset_name)+"_rBNN_ep="+str(epochs)+"_lr="+str(lr)+"_"+str(self.inference)+"/"

	# def predict(self, x, n_samples=10):

	# 	if self.inference == "svi":

	# 		sampled_models = [self.guide(None, None) for _ in range(n_samples)]
	# 		yhats = [model(x).data for model in sampled_models]
	# 		mean = torch.mean(torch.stack(yhats), 0)
	# 		return np.argmax(mean.numpy(), axis=1)

	def model(self, x_data, y_data):
		net = self.net

		if self.inference == "svi":

			for weights_name in pyro.get_param_store():
				if weights_name not in ["outw_mu","outw_sigma","outb_mu","outb_sigma"]:
					pyro.get_param_store()[weights_name].requires_grad=False

			# pyro.get_param_store()["module$$$l1.0.weight"].requires_grad=False
			# pyro.get_param_store()["module$$$l1.0.bias"].requires_grad=False
			# pyro.get_param_store()["module$$$fc2.weight"].requires_grad=False
			# pyro.get_param_store()["module$$$fc2.bias"].requires_grad=False
			# pyro.get_param_store()["module$$$fc3.weight"].requires_grad=False
			# pyro.get_param_store()["module$$$fc3.bias"].requires_grad=False

		outw_prior = Normal(loc=torch.zeros_like(net.out.weight), 
			                scale=torch.ones_like(net.out.weight))
		outb_prior = Normal(loc=torch.zeros_like(net.out.bias), 
                            scale=torch.ones_like(net.out.bias))
		
		priors = {'out.weight': outw_prior, 'out.bias': outb_prior}
		lifted_module = pyro.random_module("module", net, priors)()
		lhat = F.log_softmax(lifted_module(x_data), dim=-1)
		cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
		return cond_model

	def guide(self, x_data, y_data=None):

		net = self.net 

		outw_mu = torch.randn_like(net.out.weight)
		outw_sigma = torch.randn_like(net.out.weight)
		outw_mu_param = pyro.param("outw_mu", outw_mu)
		outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
		outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)

		outb_mu = torch.randn_like(net.out.	bias)
		outb_sigma = torch.randn_like(net.out.bias)
		outb_mu_param = pyro.param("outb_mu", outb_mu)
		outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
		outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param).independent(1)

		priors = {'out.weight': outw_prior, 'out.bias': outb_prior}
		lifted_module = pyro.random_module("module", net, priors)()
		logits = F.log_softmax(lifted_module(x_data), dim=-1)
		return logits
 
	def save(self, epochs, lr):

		path = self.get_filepath(epochs, lr)
		filename = "weights"
		os.makedirs(os.path.dirname(TESTS + path), exist_ok=True)

		if self.inference == "svi":
			param_store = pyro.get_param_store()
			print(f"\nlearned params = {param_store.get_all_param_names()}")
			param_store.save(TESTS + path + filename +".pt")

		elif self.inference == "hmc":
			save_to_pickle(data=self.posterior_samples, path=path, filename=filename+".pkl")

	def load(self, epochs, lr, rel_path=TESTS):

		path = rel_path+self.get_filepath(epochs, lr)
		filename = "weights"

		if self.inference == "svi":
			param_store = pyro.get_param_store()
			param_store.load(path + filename + ".pt")
			print("\nLoading ", path + filename + ".pt\n")

		elif self.inference == "hmc":
			posterior_samples = load_from_pickle(path + filename + ".pkl")
			self.posterior_samples = posterior_samples
			print("\nLoading ", path + filename + ".pkl\n")

	def forward(self, inputs, n_samples=10, device="cpu"):

		if self.inference == "svi":

			if DEBUG:
				print("\nguide_trace =", list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))

			preds = []
			for i in range(n_samples):
				# pyro.set_rng_seed(i) # todo: debug
				guide_trace = poutine.trace(self.guide).get_trace(inputs)			
				preds.append(guide_trace.nodes['_RETURN']['value'])

				if DEBUG:
					print("\nmodule$$$l2.0.weight shoud be fixed:\n", 
						  guide_trace.nodes['module$$$l2.0.weight']['value'][0,0,:3])
					print("\noutw_mu shoud be fixed:\n", guide_trace.nodes['outw_mu']['value'][:3])
					print("\nmodule$$$out.weight shoud change:\n", 
						  guide_trace.nodes['module$$$out.weight']['value'][0][:3]) 

		elif self.inference == "hmc":

			if DEBUG:
				print("\nself.net.state_dict() keys = ", self.net.state_dict().keys())

			preds = []
			for i in range(n_samples-1):
				state_dict = self.net.state_dict()
				out_w = self.posterior_samples["module$$$out.weight"][i]
				out_b = self.posterior_samples["module$$$out.bias"][i]
				state_dict.update({"out.weight":out_w, "out.bias":out_b})
				self.net.load_state_dict(state_dict)
				preds.append(self.net.forward(inputs))

				if DEBUG:
					print("\nl2.0.weight should be fixed:\n", 
						  self.net.state_dict()["l2.0.weight"][0,0,:3])
					print("\nout.weight should change:\n", self.net.state_dict()["out.weight"][0][:3])	
		
		pred = torch.stack(preds, dim=0)
		return pred.mean(0) 

	def _train_hmc(self, train_loader, epochs, lr, device):
		print("\n == rBNN HMC training ==")

		hmc_kernel = HMC(self.model, step_size=0.0855, num_steps=4)
		# hmc_kernel = NUTS(self.model)
		mcmc = MCMC(kernel=hmc_kernel, num_samples=10, warmup_steps=10, num_chains=1)

		start = time.time()
		n_inputs = 0
		for images, labels in train_loader:
			n_inputs += len(images)
			images = images.to(device)
			labels = labels.to(device).argmax(-1)
			mcmc.run(images.to(device), labels.to(device))

		execution_time(start=start, end=time.time())

		mcmc_samples = mcmc.get_samples()
		self.posterior_samples = mcmc_samples
		self.save(epochs=epochs, lr=lr)

		return mcmc_samples

	def _train_svi(self, train_loader, epochs, lr, device):
		print("\n == rBNN SVI training ==")
		# self.net.load(epochs=epochs, lr=lr)

		optimizer = pyro.optim.Adam({"lr": 0.001})
		elbo = TraceMeanField_ELBO()
		svi = SVI(self.model, self.guide, optimizer, loss=elbo)

		start = time.time()
		for epoch in range(epochs):
			total_loss = 0.0
			correct_predictions = 0.0
			accuracy = 0.0
			total = 0.0

			n_inputs = 0
			for images, labels in train_loader:
				n_inputs += len(images)
				images = images.to(device)
				labels = labels.to(device).argmax(-1)
				total += labels.size(0)

				outputs = self.forward(images).to(device)
				loss = svi.step(x_data=images, y_data=labels)

				predictions = outputs.argmax(dim=1)
				total_loss += loss / len(train_loader.dataset)
				correct_predictions += (predictions == labels).sum()
				accuracy = 100 * correct_predictions / len(train_loader.dataset)

			# if DEBUG:
			# 	print("\nmodule$$$l2.0.weight should be fixed:\n",
			# 		  pyro.get_param_store()["module$$$l2.0.weight"][0][0][:3])
			# 	print("\noutw_mu should change:\n", pyro.get_param_store()["outw_mu"][:3])

			print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
				  end="\t")

		execution_time(start=start, end=time.time())
		self.save(epochs=epochs, lr=lr)

	def train(self, train_loader, epochs, lr, device):

		if self.inference == "svi":
			self._train_svi(train_loader, epochs, lr, device)
		elif self.inference == "hmc":
			self._train_hmc(train_loader, epochs, lr, device)

		# todo: add loss accuracy plot

	def evaluate(self, test_loader, device):

		with torch.no_grad():

			correct_predictions = 0.0

			for images, labels in test_loader:

				images = images.to(device)
				labels = labels.to(device).argmax(-1)
				outputs = self.forward(images)
				predictions = outputs.argmax(dim=1)
				correct_predictions += (predictions == labels).sum()

			accuracy = 100 * correct_predictions / len(test_loader.dataset)
			print("\nAccuracy: %.2f%%" % (accuracy))
			return accuracy


def main(args):
	train_loader, test_loader, inp_shape, out_size = \
							data_loaders(dataset_name=args.dataset, batch_size=128, 
										 n_inputs=args.inputs, shuffle=True)

	nn = NN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size)

	nn.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, device=args.device)
	nn.load(epochs=args.epochs, lr=args.lr)
	exit()

	nn.evaluate(test_loader=test_loader, device=args.device)

	bnn = rBNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
		       inference=args.inference, base_net=nn)

	bnn.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, device=args.device)
	# bnn.load(epochs=args.epochs, n_inputs=args.n_inputs, inference=args.inference)

	bnn.evaluate(test_loader=test_loader, device=args.device)

	# n_samples_list = [5,10,50]

	# bnn.return_all_preds = True
	# for n_samples in n_samples_list:
	# 	expected_loss_gradients(posterior=bnn, n_samples=n_samples, dataset_name=args.dataset,
	# 				            model_idx=model_idx, data_loader=test_loader, device=args.device, 
	# 				            mode="vi")

	# exp_loss_gradients = []
	# for n_samples in n_samples_list:
	# 	exp_loss_gradients.append(load_loss_gradients(dataset_name=args.dataset, 
	# 						n_inputs=len(test_loader.dataset), 
	# 						n_samples=n_samples, model_idx=model_idx, relpath TESTS))

	# plot_single_images_vanishing_gradients(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
	#            fig_idx="_model="+str(0)+"_samples="+str(n_samples_list))

	# plot_exp_loss_gradients_norms(loss_gradients=exp_loss_gradients, n_inputs=args.inputs,
	# 					n_samples_list=n_samples_list,  dataset_name=args.dataset, model_idx=model_idx)

	# plot_gradient_components(n_inputs=len(test_loader.dataset), n_samples_list=n_samples_list, 
	# 	                     relpath TESTS, model_idx=model_idx, dataset_name=args.dataset,
	# 	                     loss_gradients=exp_loss_gradients)
   

if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="reduced BNN")

    parser.add_argument("--inputs", nargs="?", default=100, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--lr", nargs='?', default=0.001, type=float)
    parser.add_argument("--inference", nargs='?', default="svi", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')	

    main(args=parser.parse_args())