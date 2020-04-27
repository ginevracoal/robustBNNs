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
import torch.nn.functional as F
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.infer.mcmc.util import predictive
from pyro.infer.abstract_infer import TracePredictive
from pyro.distributions import OneHotCategorical, Normal, Categorical

softplus = torch.nn.Softplus()


DEBUG=False


class NN(nn.Module):

	def __init__(self, dataset_name, input_shape, output_size):
		super(NN, self).__init__()
		self.dataset_name = dataset_name
		self.criterion = nn.CrossEntropyLoss()
		in_channels = input_shape[0]

		if self.dataset_name == "mnist":
			self.l1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5),
									nn.LeakyReLU(),
									nn.MaxPool2d(kernel_size=2))
			self.l2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5),
									nn.LeakyReLU(),
									nn.MaxPool2d(kernel_size=2),#)
									nn.Flatten())
			self.out = nn.Linear(4*4*64, output_size)
		else:
			raise NotImplementedError()

	def get_filename(self, epochs, lr):
		self.filename = str(self.dataset_name)+"_nn_ep="+str(epochs)+"_lr="+str(lr)
		return self.filename

	def forward(self, inputs):
		x = inputs

		x = self.l1(x)
		x = self.l2(x)
		# x = x.view(x.size(0), -1)
		x = self.out(x)

		output = nn.Softmax(dim=-1)(x)
		return output

	def save(self, epochs, lr):
		filename = self.get_filename(epochs, lr)
		os.makedirs(os.path.dirname(TESTS)+filename+"/", exist_ok=True)
		print("\nSaving: ", TESTS+filename+"/"+filename+"_weights.pt")
		torch.save(self.state_dict(), TESTS+filename+"/"+filename+"_weights.pt")

		if DEBUG:
			print("\nCheck saved weights:")
			print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
			print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

	def load(self, epochs, lr, device, rel_path=TESTS):
		filename = self.get_filename(epochs, lr)
		print("\nLoading: ", rel_path+filename+"/"+filename+"_weights.pt")
		self.load_state_dict(torch.load(rel_path+filename+"/"+filename+"_weights.pt"))
		# print("\n", list(self.state_dict().keys()), "\n")
		self.to(device)

		if DEBUG:
			print("\nCheck loaded weights:")	
			print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
			print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

	def train(self, train_loader, epochs, lr, device):
		print("\n == NN training ==")
		random.seed(0)
		self.to(device)

		optimizer = torchopt.Adam(params=self.parameters(), lr=lr)

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

				optimizer.zero_grad()
				outputs = self.forward(x_batch)
				loss = self.criterion(outputs, y_batch)
				loss.backward()
				optimizer.step()

				predictions = outputs.argmax(dim=1)
				total_loss += loss.data.item() / len(train_loader.dataset)
				correct_predictions += (predictions == y_batch).sum()
				accuracy = 100 * correct_predictions / len(train_loader.dataset)

			# print(self.l2.0.weight.data)

			print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
				  end="\t")

		execution_time(start=start, end=time.time())
		self.save(epochs=epochs, lr=lr)

	def evaluate(self, test_loader, device):
		self.to(device)
		random.seed(0)

		with torch.no_grad():

			correct_predictions = 0.0

			for x_batch, y_batch in test_loader:

				x_batch = x_batch.to(device)
				y_batch = y_batch.to(device).argmax(-1)
				outputs = self(x_batch)
				predictions = outputs.argmax(dim=1)
				correct_predictions += (predictions == y_batch).sum()

			accuracy = 100 * correct_predictions / len(test_loader.dataset)
			print("\nAccuracy: %.2f%%" % (accuracy))
			return accuracy


class redBNN(nn.Module):

	def __init__(self, dataset_name, input_shape, output_size, inference, base_net):
		super(redBNN, self).__init__()
		self.dataset_name = dataset_name
		self.inference = inference
		self.net = base_net

	def get_hyperparams(self, args):

		if self.inference == "svi":
			return {"epochs":args.epochs, "lr":args.lr}

		elif self.inference == "hmc":
			return {"hmc_samples":args.hmc_samples, "warmup":args.warmup}

	def get_filename(self, n_inputs, hyperparams):

		if self.inference == "svi":
			return str(self.dataset_name)+"_redBNN_inp="+str(n_inputs)+"_ep="+\
			       str(hyperparams["epochs"])+"_lr="+str(hyperparams["lr"])+"_"+\
			       str(self.inference)

		elif self.inference == "hmc":
			return str(self.dataset_name)+"_redBNN_inp="+str(n_inputs)+"_samp="+\
			       str(hyperparams["hmc_samples"])+"_warm="+str(hyperparams["warmup"])+\
			       "_"+str(self.inference)

	def model(self, x_data, y_data):
		net = self.net

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
 
	def save(self, n_inputs, hyperparams):

		path = TESTS + self.get_filename(n_inputs, hyperparams)+"/"
		filename = self.get_filename(n_inputs, hyperparams)+"_weights"
		os.makedirs(os.path.dirname(path), exist_ok=True)

		if self.inference == "svi":
			self.net.to("cpu")
			self.to("cpu")
			param_store = pyro.get_param_store()
			print("\nSaving: ", path + filename +".pt")
			print(f"\nlearned params = {param_store.get_all_param_names()}")
			param_store.save(path + filename +".pt")

		elif self.inference == "hmc":
			self.net.to("cpu")
			self.to("cpu")
			save_to_pickle(data=self.posterior_samples, path=path, filename=filename+".pkl")

	def load(self, n_inputs, hyperparams, device, rel_path=TESTS):

		path = rel_path+self.get_filename(n_inputs, hyperparams)+"/"
		filename = self.get_filename(n_inputs, hyperparams)+"_weights"

		if self.inference == "svi":
			param_store = pyro.get_param_store()
			param_store.load(path + filename + ".pt")
			print("\nLoading ", path + filename + ".pt\n")

		elif self.inference == "hmc":
			posterior_samples = load_from_pickle(path + filename + ".pkl")
			self.posterior_samples = posterior_samples
			print("\nLoading ", path + filename + ".pkl\n")

		self.to(device)
		self.net.to(device)

	def forward(self, inputs, n_samples=100):

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
			n_samples = min(n_samples, len(self.posterior_samples["module$$$out.weight"]))
			for i in range(n_samples):
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
		
		print(f"{n_samples} post samp", end="\t")
		pred = torch.stack(preds, dim=0)
		return pred.mean(0) 

	def _train_hmc(self, train_loader, hyperparams, device):
		print("\n == redBNN HMC training ==")

		num_samples, warmup_steps = (hyperparams["hmc_samples"], hyperparams["warmup"])

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

	def _train_svi(self, train_loader, hyperparams, device):
		print("\n == redBNN SVI training ==")

		epochs, lr = hyperparams["epochs"], hyperparams["lr"]

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
		self.save(n_inputs=len(train_loader.dataset), hyperparams=hyperparams)

	def train(self, train_loader, hyperparams, device):
		self.to(device)
		self.net.to(device)
		random.seed(0)
		pyro.set_rng_seed(0)

		if self.inference == "svi":
			self._train_svi(train_loader, hyperparams, device)

		elif self.inference == "hmc":
			self._train_hmc(train_loader, hyperparams, device)

	def evaluate(self, test_loader, device):
		self.to(device)
		self.net.to(device)
		random.seed(0)
		pyro.set_rng_seed(0)

		with torch.no_grad():

			correct_predictions = 0.0

			for x_batch, y_batch in test_loader:

				x_batch = x_batch.to(device)
				y_batch = y_batch.to(device).argmax(-1)
				outputs = self.forward(x_batch)
				predictions = outputs.argmax(dim=1)
				correct_predictions += (predictions == y_batch).sum()

			accuracy = 100 * correct_predictions / len(test_loader.dataset)
			print("\nAccuracy: %.2f%%" % (accuracy))
			return accuracy


def main(args):

	# === load dataset ===
	train_loader, test_loader, inp_shape, out_size = \
							data_loaders(dataset_name=args.dataset, batch_size=256, 
										 n_inputs=args.inputs, shuffle=True)
	# === base NN ===

	# dataset, epochs, lr, rel_path = (args.dataset, args.epochs, args.lr, TESTS)		
	dataset, epochs, lr, rel_path = ("mnist", 20, 0.001, DATA)	

	nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size)
	# nn.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, device=args.device)
	nn.load(epochs=epochs, lr=lr, device=args.device, rel_path=rel_path)
	nn.evaluate(test_loader=test_loader, device=args.device)

	# === reducedBNN ===
	bnn = redBNN(dataset_name=args.dataset, input_shape=inp_shape, output_size=out_size, 
		         inference=args.inference, base_net=nn)
	hyperparams = bnn.get_hyperparams(args)
	bnn.train(train_loader=train_loader, hyperparams=hyperparams, device=args.device)
	# bnn.load(n_inputs=args.inputs, hyperparams=hyperparams, device=args.device, rel_path=TESTS)
	bnn.evaluate(test_loader=test_loader, device=args.device)
	

if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="reduced BNN")

    parser.add_argument("--inputs", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist, fashion_mnist, cifar")
    parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--hmc_samples", default=5, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")	

    main(args=parser.parse_args())