import argparse
import os
from directories import *
from utils import *
import pyro
import torch
from torch import nn
import torch.nn.functional as nnf
import numpy as np
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
import torch.optim as torchopt
from pyro import poutine
import pyro.optim as pyroopt
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.infer.mcmc.util import predictive
from pyro.infer.abstract_infer import TracePredictive
from pyro.distributions import OneHotCategorical, Normal, Categorical
from nn import NN
import pandas

from gp_reduced_bnn import attack_increasing_eps, plot_increasing_eps

softplus = torch.nn.Softplus()
DEBUG=False


saved_redBNNs = {"model_0":{"dataset":"mnist", "inference":"svi", "hidden_size":64, 
                 			"base_inputs":30000, "base_epochs":15, "base_lr":0.001,
                 			"bnn_inputs":10000, "bnn_epochs":30, "bnn_lr":0.01, 
                 			# "bnn_inputs":60000, "bnn_epochs":30, "bnn_lr":0.01, 
                 			"activation":"leaky", "architecture":"fc2"}}


class baseNN(NN):

	def __init__(self, dataset_name, input_shape, output_size, hidden_size, activation, 
                       architecture, epochs, lr):
		super(baseNN, self).__init__(dataset_name, input_shape, output_size, hidden_size, activation, 
                                 architecture)
		self.epochs = epochs
		self.lr = lr
		self.savedir = str(dataset_name)+"_RedBNN_hid="+str(hidden_size)+\
		                    "_ep="+str(self.epochs)+"_lr="+str(self.lr)
		self.name = str(dataset_name)+"_baseRedNN_hid="+str(hidden_size)+\
		                    "_ep="+str(self.epochs)+"_lr="+str(self.lr)

	def set_model(self, architecture, activation, input_shape, output_size, hidden_size):

		input_size = input_shape[0]*input_shape[1]*input_shape[2]
		in_channels = input_shape[0]

		if activation == "relu":
			activ = nn.ReLU
		elif activation == "leaky":
			activ = nn.LeakyReLU
		elif activation == "sigm":
			activ = nn.Sigmoid
		elif activation == "tanh":
			activ = nn.Tanh
		else: 
			raise AssertionError("\nWrong activation name.")

		### separating the last layer from the others
		if architecture == "fc":
			self.model = nn.Sequential(
				nn.Flatten(), 
				nn.Linear(input_size, hidden_size),
				activ())
			self.out = nn.Linear(hidden_size, output_size)

		elif architecture == "fc2":
			self.model = nn.Sequential(
				nn.Flatten(),
				nn.Linear(input_size, hidden_size),
				activ(),
				nn.Linear(hidden_size, hidden_size),
				activ())
			self.out = nn.Linear(hidden_size, output_size)

		elif architecture == "conv":
			self.model = nn.Sequential(
				nn.Conv2d(in_channels, 32, kernel_size=5),
				activ(),
				nn.MaxPool2d(kernel_size=2),
				nn.Conv2d(32, hidden_size, kernel_size=5),
				activ(),
				nn.MaxPool2d(kernel_size=2, stride=1),
				nn.Flatten())
			self.out = nn.Linear(int(hidden_size/(4*4))*input_shape, output_size)
		else:
			raise NotImplementedError()

	def train(self, train_loader, device):
		super(baseNN, self).train(train_loader=train_loader, epochs=self.epochs, 
			                      lr=self.lr, device=device)

	def forward(self, inputs, train=False):
		x = self.model(inputs)
		x = self.out(x)
		return nnf.log_softmax(x, dim=-1) if train is True else nnf.softmax(x, dim=-1)

	def save(self, epochs=None, lr=None):
		filepath, filename = (TESTS+self.savedir+"/", self.name+"_weights.pt")
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		print("\nSaving: ", filepath+filename)
		torch.save(self.state_dict(),filepath+filename)

		if DEBUG:
			print("\nCheck saved weights:")
			print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
			print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

	def load(self, rel_path=TESTS, epochs=None, lr=None):
		filepath, filename = (TESTS+self.savedir+"/", self.name+"_weights.pt")
		print("\nLoading: ", filepath+filename)
		self.load_state_dict(torch.load(filepath+filename))
		# print("\n", list(self.state_dict().keys()), "\n")
		# self.to(device)

		if DEBUG:
			print("\nCheck loaded weights:")	
			print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
			print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])


def get_hyperparams(model_dict):

	if model_dict["inference"] == "svi":
		return {"epochs":model_dict["bnn_epochs"], "lr":model_dict["bnn_lr"]}

	elif model_dict["inference"] == "hmc":
		return {"hmc_samples":model_dict["hmc_samples"], "warmup":model_dict["warmup"]}


class redBNN(nn.Module):

	def __init__(self, dataset_name, inference, hyperparams, base_net):
		super(redBNN, self).__init__()
		self.dataset_name = dataset_name
		self.inference = inference
		self.base_net = base_net
		self.hyperparams = hyperparams

	def get_filename(self, n_inputs):

		if self.inference == "svi":
			return str(self.dataset_name)+"_redBNN_inp="+str(n_inputs)+"_ep="+\
			       str(self.hyperparams["epochs"])+"_lr="+str(self.hyperparams["lr"])+"_"+\
			       str(self.inference)

		elif self.inference == "hmc":
			return str(self.dataset_name)+"_redBNN_inp="+str(n_inputs)+"_samp="+\
			       str(self.hyperparams["hmc_samples"])+"_warm="+str(self.hyperparams["warmup"])+\
			       "_"+str(self.inference)

	def model(self, x_data, y_data):
		net = self.base_net

		if self.inference == "svi":
			for weights_name in pyro.get_param_store():
				if weights_name not in ["outw_mu","outw_sigma","outb_mu","outb_sigma"]:
					pyro.get_param_store()[weights_name].requires_grad=False

		outw_prior = Normal(loc=torch.zeros_like(net.out.weight), 
			                scale=torch.ones_like(net.out.weight))
		outb_prior = Normal(loc=torch.zeros_like(net.out.bias), 
                            scale=torch.ones_like(net.out.bias))
		
		priors = {'out.weight': outw_prior, 'out.bias': outb_prior}
		lifted_module = pyro.random_module("module", net, priors)()
		lhat = nnf.log_softmax(lifted_module(x_data), dim=-1)
		cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
		return cond_model

	def guide(self, x_data, y_data=None):
		net = self.base_net 

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
		logits = nnf.log_softmax(lifted_module(x_data), dim=-1)
		return logits
 
	def save(self, n_inputs):

		filepath, filename = (TESTS+self.base_net.savedir+"/", self.get_filename(n_inputs)+"_weights")
		os.makedirs(os.path.dirname(filepath), exist_ok=True)

		if self.inference == "svi":
			self.base_net.to("cpu")
			self.to("cpu")
			param_store = pyro.get_param_store()
			print("\nSaving: ", filepath + filename +".pt")
			print(f"\nlearned params = {param_store.get_all_param_names()}")
			param_store.save(filepath + filename +".pt")

		elif self.inference == "hmc":
			self.base_net.to("cpu")
			self.to("cpu")
			save_to_pickle(data=self.posterior_samples, path=filepath, filename=filename+".pkl")

	def load(self, n_inputs, device, rel_path=TESTS):

		filepath, filename = (TESTS+self.base_net.savedir+"/", self.get_filename(n_inputs)+"_weights")

		if self.inference == "svi":
			param_store = pyro.get_param_store()
			param_store.load(filepath + filename + ".pt")
			print("\nLoading ", filepath + filename + ".pt\n")

		elif self.inference == "hmc":
			posterior_samples = load_from_pickle(filepath + filename + ".pkl")
			self.posterior_samples = posterior_samples
			print("\nLoading ", filepath + filename + ".pkl\n")

		# self.to(device)
		self.base_net.to(device)

	def forward(self, inputs, n_samples=100):

		if self.inference == "svi":

			if DEBUG:
				print("\nguide_trace =", 
					list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))

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
				print("\nself.base_net.state_dict() keys = ", self.base_net.state_dict().keys())

			preds = []
			n_samples = min(n_samples, len(self.posterior_samples["module$$$out.weight"]))
			for i in range(n_samples):
				state_dict = self.base_net.state_dict()
				out_w = self.posterior_samples["module$$$out.weight"][i]
				out_b = self.posterior_samples["module$$$out.bias"][i]
				state_dict.update({"out.weight":out_w, "out.bias":out_b})
				self.base_net.load_state_dict(state_dict)
				preds.append(self.base_net.forward(inputs))

				if DEBUG:
					print("\nl2.0.weight should be fixed:\n", 
						  self.base_net.state_dict()["l2.0.weight"][0,0,:3])
					print("\nout.weight should change:\n", self.base_net.state_dict()["out.weight"][0][:3])	
		
		stacked_preds = torch.stack(preds, dim=0)
		return nnf.softmax(stacked_preds.mean(0), dim=-1)

	def _train_hmc(self, train_loader, device):
		print("\n == redBNN HMC training ==")

		num_samples, warmup_steps = (self.hyperparams["hmc_samples"], self.hyperparams["warmup"])

		# kernel = HMC(self.model, step_size=0.0855, num_steps=4)
		kernel = NUTS(self.model)
		batch_samples = int(num_samples*train_loader.batch_size/len(train_loader.dataset))+1
		print("\nSamples per batch =", batch_samples)
		hmc = MCMC(kernel=kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)

		start = time.time()

		out_weight, out_bias = ([],[])
		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device).argmax(-1)
			hmc.run(x_batch, y_batch)
			out_weight.append(hmc.get_samples()["module$$$out.weight"])
			out_bias.append(hmc.get_samples()["module$$$out.bias"])

		execution_time(start=start, end=time.time())

		out_weight, out_bias = (torch.cat(out_weight), torch.cat(out_bias))
		self.posterior_samples = {"module$$$out.weight":out_weight, "module$$$out.bias":out_bias}

		self.save(n_inputs=len(train_loader.dataset), hyperparams=hyperparams)
		return self.posterior_samples

	def _train_svi(self, train_loader, device):
		print("\n == redBNN SVI training ==")

		epochs, lr = (self.hyperparams["epochs"], self.hyperparams["lr"])

		optimizer = pyro.optim.Adam({"lr":lr})
		elbo = TraceMeanField_ELBO()
		svi = SVI(self.model, self.guide, optimizer, loss=elbo)

		start = time.time()
		for epoch in range(epochs):
			total_loss = 0.0
			correct_predictions = 0.0
			accuracy = 0.0
			total = 0.0

			n_inputs = 0
			for x_batch, y_batch in train_loader:
				n_inputs += len(x_batch)
				x_batch = x_batch.to(device)
				y_batch = y_batch.to(device).argmax(-1)
				total += y_batch.size(0)

				outputs = self.forward(x_batch).to(device)
				loss = svi.step(x_data=x_batch, y_data=y_batch)

				predictions = outputs.argmax(dim=1)
				total_loss += loss / len(train_loader.dataset)
				correct_predictions += (predictions == y_batch).sum()
				accuracy = 100 * correct_predictions / len(train_loader.dataset)

			# if DEBUG:
			# 	print("\nmodule$$$l2.0.weight should be fixed:\n",
			# 		  pyro.get_param_store()["module$$$l2.0.weight"][0][0][:3])
			# 	print("\noutw_mu should change:\n", pyro.get_param_store()["outw_mu"][:3])

			print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
				  end="\t")

		execution_time(start=start, end=time.time())
		hyperparams = {"epochs":epochs, "lr":lr}	
		self.save(n_inputs=len(train_loader.dataset))

	def train(self, train_loader, device):
		self.to(device)
		self.base_net.to(device)
		random.seed(0)
		pyro.set_rng_seed(0)

		if self.inference == "svi":
			self._train_svi(train_loader, device)

		elif self.inference == "hmc":
			self._train_hmc(train_loader, device)

	def evaluate(self, test_loader, device, n_samples=10):
		self.to(device)
		self.base_net.to(device)
		random.seed(0)
		pyro.set_rng_seed(0)

		with torch.no_grad():

			correct_predictions = 0.0

			for x_batch, y_batch in test_loader:

				x_batch = x_batch.to(device)
				y_batch = y_batch.to(device).argmax(-1)
				outputs = self.forward(x_batch, n_samples=n_samples)
				predictions = outputs.argmax(dim=1)
				correct_predictions += (predictions == y_batch).sum()

			accuracy = 100 * correct_predictions / len(test_loader.dataset)
			print("\nAccuracy: %.2f%%" % (accuracy))
			return accuracy


def main(args):

	m = saved_redBNNs["model_0"]

	if args.device=="cuda":
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	# === base NN ===
	train_loader, test_loader, inp_shape, out_size = \
							data_loaders(dataset_name=m["dataset"], batch_size=64, 
										 n_inputs=m["base_inputs"], shuffle=True)

	nn = baseNN(dataset_name=m["dataset"], input_shape=inp_shape, output_size=out_size,
		        epochs=m["base_epochs"], lr=m["base_lr"], hidden_size=m["hidden_size"], 
		        activation=m["activation"], architecture=m["architecture"])
	# nn.train(train_loader=train_loader, device=args.device)
	nn.load(rel_path=TESTS)
	nn.evaluate(test_loader=test_loader, device=args.device)

	# === reducedBNN ===
	train_loader, test_loader, inp_shape, out_size = \
							data_loaders(dataset_name=m["dataset"], batch_size=128, 
										 n_inputs=m["bnn_inputs"], shuffle=True)
	hyp = get_hyperparams(m)

	bnn = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=nn, hyperparams=hyp)
	# bnn.train(train_loader=train_loader, device=args.device)
	bnn.load(n_inputs=m["bnn_inputs"], device=args.device, rel_path=TESTS)
	bnn.evaluate(test_loader=test_loader, device=args.device)
	
	# === multiple attacks ===
	bnn_samples = 1000
	# df = attack_increasing_eps(nn=nn, bnn=bnn, dataset=m["dataset"], device=args.device, 
	# 	                       method=args.attack, n_samples=bnn_samples)
	df = pandas.read_csv(TESTS+str(m["dataset"])+"_increasing_eps_"+str(args.attack)+".csv")
	plot_increasing_eps(df, dataset=m["dataset"], method=args.attack, n_samples=bnn_samples)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="reduced BNN")
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")	
    parser.add_argument("--attack", default="fgsm", type=str, help="fgsm, pgd")
    main(args=parser.parse_args())