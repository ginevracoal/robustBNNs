import sys
sys.path.append(".")
from directories import *
from lossGradients import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from bnn import saved_bnns
from utils import classwise_data_loaders


def plot_gradients_components_classes(loss_gradients, n_inputs, n_samples_list, name):
    matplotlib.rc('font', **{'weight':'bold', 'size': 8})
    fig, ax = plt.subplots(2, 5, figsize=(11, 5), dpi=150, facecolor='w', edgecolor='k')
    sns.set_palette("gist_heat", 5)
    
    for label in range(10):
        print("\nlabel =", label, end="\n")
        plot_samples = []
        loss_gradients_components = []
        for samples_idx, n_samples in enumerate(n_samples_list):
            samples_grad_components = np.array(loss_gradients[samples_idx, label]).flatten()
            loss_gradients_components.extend(samples_grad_components)
            plot_samples.extend(np.repeat(n_samples, len(samples_grad_components)))

        df = pd.DataFrame(data={"loss_gradients": loss_gradients_components, 
                                "n_samples": plot_samples})
        print(df.describe())

        axis = ax[0, label] if label < 5 else ax[1, label-5]
        
        sns.stripplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, 
                      ax=axis, jitter=0.2, alpha=0.4)

        axis.set_title("label = "+str(label), fontdict={"fontsize":9,"fontweight":"bold"},
            pad=0.5)
        axis.set_ylabel("")
        axis.set_xlabel("")
        # axis.set_yscale('log')

    fig.text(0.5, 0., "Samples involved in the expectations ($w \sim p(w|D)$)", 
             ha='center', size=10)
    fig.text(0., 0.5, r"Expected Gradients components $\langle\nabla L(x,w)\rangle_{w}$", 
             va='center', rotation='vertical', size=10)

    plt.tight_layout()
    path = TESTS+name+"/"
    filename = "expLossGrads_inp="+str(n_inputs)+"_stripplot.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path+filename)


def main(args):

    _, test_loaders, inp_shape, out_size = \
     classwise_data_loaders(dataset_name=args.dataset, batch_size=64, n_inputs=args.inputs)

    # === load BNN ===
    hidden_size, activation, architecture, inference, \
    epochs, lr, samples, warmup = saved_bnns[args.dataset]

    bnn = BNN(args.dataset, hidden_size, activation, architecture, inference,
              epochs, lr, samples, warmup, inp_shape, out_size)
    bnn.load(device=args.device, rel_path=DATA)

    n_samples_list = [1,10,50,100,500]

    for class_idx, test_loader in enumerate(test_loaders):
        print(f"\nlabel={class_idx}")
        for n_samples in n_samples_list:
            print(f"samples={n_samples}" , end="\t")
            bnn.evaluate(test_loader=test_loader, device=args.device, n_samples=n_samples)

    # === compute gradients === #

    # for n_samples in n_samples_list:
    #     for class_idx, test_loader in enumerate(test_loaders):

    #         savedir, filename = (bnn.name+"/classwise_gradients/", "class="+str(class_idx))
    #         loss_gradients(net=bnn, n_samples=n_samples, filename=filename, savedir=savedir,
    #                        data_loader=test_loader, device=args.device)

    # === load and plot === #

    gradients = []
    for n_samples in n_samples_list:
        samples_gradients = []
        for class_idx in range(out_size):
            savedir, filename = (bnn.name+"/classwise_gradients/", "class="+str(class_idx))
            samples_gradients.append(load_loss_gradients(n_samples=n_samples, 
                              filename=filename, relpath=TESTS, savedir=savedir))
        gradients.append(samples_gradients)

    plot_gradients_components_classes(loss_gradients=np.array(gradients), 
                                      n_inputs=args.inputs,
                                      n_samples_list=n_samples_list, name=bnn.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=50, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, 
                        help="mnist, fashion_mnist, cifar")
    parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
    main(args=parser.parse_args())