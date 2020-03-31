import argparse
import os
from directories import *
from utils import *
import pyro
import torch
from torch import nn
from reducedBNN import NN
import torch.nn.functional as nnf
import numpy as np
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
import torch.optim as torc hopt
from pyro import poutine
import pyro.optim as pyroopt
import torch.nn.functional as F
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.infer.mcmc.util import predictive
from pyro.infer.abstract_infer import TracePredictive
from pyro.distributions import OneHotCategorical, Normal, Categorical
from lossGradients import loss_gradient

softplus = torch.nn.Softplus()


DEBUG=False


def regularized_cross_entropy(net, x, y, lam):
	# y = one hot encoding
	log_prob = -1.0 * F.log_softmax(x, 1)
	loss = log_prob.gather(1, y.unsqueeze(1))
	loss = loss.mean()
	gradient = nn_loss_gradient(net, image, label)
	loss_gradient_norm = np.max(np.abs(gradient))
	reg_loss = loss + lam*loss_gradient_norm
	return reg_loss


class BRNN(nn.Module):

	def __init__(self, dataset_name, input_shape, output_size, inference, lam=0.5):
		self.base_net = NN(dataset_name=dataset_name, input_shape=input_shape, output_size=output_size)
		self.dataset_name = dataset_name
		self.inference = inference
		self.lam = lam
		self.criterion = regularized_cross_entropy()

	def get_hyperparams(self, args):

		hyperparams = {"sgd_epochs":args.sgd_epochs, "sgd_lr":sgd_lr}

		if self.inference == "svi":
			return hyperparams.update({"svi_epochs":args.svi_epochs, "svi_lr":args.svi_lr})

		elif self.inference == "mcmc":
			return hyperparams.update({"mcmc_samples":args.mcmc_samples, "warmup":args.warmup})

	def get_filename(self, n_inputs, hyperparams):

		filename = str(self.dataset_name)+"_BRN_inp="+str(n_inputs)"_lam="+str(self.lam)+"_ep="+\
			       str(self.epochs)+"_lr="+str(self.lr)

		if self.inference == "svi":
			return filename+"_epB="+str(hyperparams["svi_epochs"])+\
       			    "_lrB="+str(hyperparams["svi_lr"])+"_"+str(self.inference)

		elif self.inference == "mcmc":
			return filename+"_samp="+str(hyperparams["mcmc_samples"])+\
			       "_warm="+str(hyperparams["warmup"])+"_"+str(self.inference)

	def model(self, x_data, y_data):

		if self.inference == "svi":
			for weights_name in pyro.get_param_store():
				if weights_name not in ["outw_mu","outw_sigma","outb_mu","outb_sigma"]:
					pyro.get_param_store()[weights_name].requires_grad=False

		outw_prior = Normal(loc=torch.zeros_like(self.out.weight), 
			                scale=torch.ones_like(self.out.weight))
		outb_prior = Normal(loc=torch.zeros_like(self.out.bias), 
                            scale=torch.ones_like(self.out.bias))
		
		priors = {'out.weight': outw_prior, 'out.bias': outb_prior}
		lifted_module = pyro.random_module("module", self, priors)()
		lhat = F.log_softmax(lifted_module(x_data), dim=-1)
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
		logits = F.log_softmax(lifted_module(x_data), dim=-1)
		return logits
 
	def save(self, n_inputs, hyperparams):

		path = TESTS + self.get_filename(n_inputs, hyperparams)+"/"
		filename = self.get_filename(n_inputs, hyperparams)+"_weights"
		os.makedirs(os.path.dirname(path), exist_ok=True)

		if self.inference == "svi":
			self.base_net.to("cpu")
			self.to("cpu")
			param_store = pyro.get_param_store()
			print("\nSaving: ", path + filename +".pt")
			print(f"\nlearned params = {param_store.get_all_param_names()}")
			param_store.save(path + filename +".pt")

		elif self.inference == "mcmc":
			self.base_net.to("cpu")
			self.to("cpu")
			save_to_pickle(data=self.posterior_samples, path=path, filename=filename+".pkl")

	def load(self, n_inputs, hyperparams, device, rel_path=TESTS):

		path = rel_path+self.get_filename(n_inputs, hyperparams)+"/"
		filename = self.get_filename(n_inputs, hyperparams)+"_weights"

		if self.inference == "svi":
			param_store = pyro.get_param_store()
			param_store.load(path + filename + ".pt")
			print("\nLoading ", path + filename + ".pt\n")

		elif self.inference == "mcmc":
			posterior_samples = load_from_pickle(path + filename + ".pkl")
			self.posterior_samples = posterior_samples
			print("\nLoading ", path + filename + ".pkl\n")

		self.to(device)
		self.base_net.to(device)

	def forward(self, inputs, n_samples=30):

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

		elif self.inference == "mcmc":

			if DEBUG:
				print("\nself.base_net.state_dict() keys = ", self.base_net.state_dict().keys())

			preds = []
			for i in range(n_samples-1):
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
		
		pred = torch.stack(preds, dim=0)
		return pred.mean(0) 

	def _train_mcmc(self, images, labels, hyperparams):
		print("\n== MCMC step ==")

		# kernel = HMC(self.model, step_size=0.0855, num_steps=4)
		kernel = NUTS(self.model)

		mcmc = MCMC(kernel=kernel, num_samples=hyperparams["mcmc_samples"], 
			        warmup_steps=hyperparams["mcmc_warmup"], num_chains=1)
		mcmc.run(images, labels)

		mcmc_samples = mcmc.get_samples()
		self.posterior_samples = mcmc_samples
		return mcmc_samples

	def _train_svi(self, images, labels, hyperparams):
		print("\n== SVI step ==")

		optimizer = pyro.optim.Adam({"lr":hyperparams["svi_lr"]})
		elbo = TraceMeanField_ELBO()
		svi = SVI(self.model, self.guide, optimizer, loss=elbo)

		loss = svi.step(x_data=images, y_data=labels)
		total_loss += loss / len(images)

		# veficare come vengono aggiornati i pesi

	def _train_sgd(self, images, labels, lr):
		print("\n --- SGD step --")

		optimizer = torchopt.Adam(params=self.net.parameters(), lr=lr)
		optimizer.zero_grad()
		outputs = self.forward(images)

		loss = self.criterion(self.net, outputs, labels, self.lam)
		loss.backward()
		optimizer.step()

		# print(self.l2.0.weight.data)

	def train(self, train_loader, hyperparams, device):
		self.to(device)
		self.base_net.to(device)
		pyro.set_rng_seed(0)

		start = time.time()
		for epoch in range(hyperparams["sgd_epochs"]):
			total_loss = 0.0
			correct_predictions = 0.0
			accuracy = 0.0

			for x_train, y_train in train_loader:

				x_train = x_train.to(device)
				y_train = y_train.to(device)
				half_batch = int(len(x_train)*0.5)

				images = x_train[:half_batch]
				labels = y_train[:half_batch]
				self._train_sgd(images, labels, lr=hyperparams["sgd_lr"])

				images = x_train[half_batch:]
				labels = y_train[half_batch:]
				if self.inference == "svi":
					self._train_svi(images, labels, hyperparams)
				elif self.inference == "mcmc":
					self._train_mcmc(images, labels, hyperparams)

				outputs = self.forward(x_train).to(device)
				predictions = outputs.argmax(dim=1)
				correct_predictions += (predictions == labels).sum()
				accuracy = 100 * correct_predictions / len(x_train)

				print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
					  end="\t")

		# campiono dalla posterior e faccio un update di self.parameters()
		execution_time(start=start, end=time.time())

		# salvo i parametri bayesiani
		self.save(n_inputs=len(train_loader.dataset), hyperparams=bayesian_hyperparams)


	def evaluate(self, test_loader, device):
		self.to(device)
		self.base_net.to(device)

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

	# === load dataset ===
	train_loader, test_loader, inp_shape, out_size = \
							data_loaders(dataset_name=args.dataset, batch_size=512, 
										 n_inputs=args.inputs, shuffle=True)

	brn = BRNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
		         inference=args.inference, lam=args.lam)
	hyperparams = brn.get_hyperparams(args)
	brn.train(train_loader=train_loader, hyperparams=hyperparams, device=args.device)
	# brn.load(n_inputs=args.inputs, bayesian_hyperparams=hyperparams, device=args.device, rel_path=TESTS)
	brn.evaluate(test_loader=test_loader, device=args.device)
	

if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')

    parser.add_argument("--inputs", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help='mnist, cifar, fashion_mnist')
    parser.add_argument("--sgd_epochs", nargs='?', default=10, type=int)
    parser.add_argument("--sgd_lr", default=0.001, type=float)
    parser.add_argument("--lam", default=0.5, type=float)
    parser.add_argument("--inference", default="svi", type=str, help='svi, mcmc')
    parser.add_argument("--mcmc_samples", default=30, type=int)
    parser.add_argument("--mcmc_warmup", default=10, type=int)
    parser.add_argument("--svi_epochs", default=10, type=int)
    parser.add_argument("--svi_lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help='cpu, cuda')	

    main(args=parser.parse_args())