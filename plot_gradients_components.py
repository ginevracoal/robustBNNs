"""
Plot gradients components and vanishing gradients heatmaps 
for an increasing number of posterior samples.
"""

from savedir import *
from lossGradients import *
from model_bnn import saved_BNNs
from utils import load_dataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def stripplot_gradients_components(loss_gradients_list, n_samples_list, dataset_name, filename, relpath):

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150, facecolor='w', edgecolor='k')    

    loss_gradients_components = []
    plot_samples = []
    for samples_idx, n_samples in enumerate(n_samples_list):
        
        print("\nsamples = ", n_samples, end="\t")
        print(f"min = {loss_gradients_list[samples_idx].min():.4f}", end="\t")
        print(f"max = {loss_gradients_list[samples_idx].max():.4f}")

        avg_loss_gradient = np.array(loss_gradients_list[samples_idx]).flatten()
        loss_gradients_components.extend(avg_loss_gradient)
        plot_samples.extend(np.repeat(n_samples, len(avg_loss_gradient)))

    df = pd.DataFrame(data={"loss_gradients": loss_gradients_components, 
                            "n_samples": plot_samples})

    sns.stripplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, ax=ax, 
                  jitter=0.2, alpha=0.4, palette="rocket")
    fig.subplots_adjust(hspace=1, wspace=0.5)
    fig.tight_layout()

    ax.set_ylabel(r"Expected gradients $\langle\frac{\partial L}{\partial x_i}(x,w)\rangle_{p(w|D)}$", weight='bold')
    ax.set_xlabel("Number of posterior samples $w \sim p(w|D)$", weight='bold')

    # fig.text(0.5, 0.01, "Number of posterior samples $w \sim p(w|D)$", ha='center')
    # fig.text(0.03, 0.5, r"Expected gradients $\langle\frac{\partial L}{\partial x_i}(x,w)\rangle_{p(w|D)}$", 
    #          va='center', rotation='vertical')

    path = relpath+filename+"/"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + filename + "_gradComponents.png")

def _vanishing_gradient_heatmap(image, gradients, n_samples_list, norm):

    fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list)+1, figsize=(11, 3))

    matplotlib.rc('font', **{'size': 10, 'weight':'bold'})
    bottom, width, height = (.10, .01, .75)

    sns.heatmap(image, ax=axs[0], square=True, cmap="Greys_r", cbar=False)
    
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    cmap = "rocket_r"
    vmin, vmax = (np.min(gradients), np.max(gradients))
    for col_idx, samples in enumerate(n_samples_list):
        loss_gradient = gradients[col_idx]
        
        cbar_ax = fig.add_axes([.93, bottom, width, height])
        sns.heatmap(loss_gradient, ax=axs[col_idx+1], square=True, cmap=cmap,
                    vmin=vmin, vmax=vmax, cbar_ax=cbar_ax,
                    cbar=True if col_idx+1==len(n_samples_list) else False)

        if norm == "linfty":
            grad_norm = np.max(np.abs(loss_gradient))
            expr = r"  $|\langle\frac{\partial L}{\partial x_i}(x,w)\rangle_{p(w|D)}|_\infty$"

        elif norm == "l2":
            grad_norm = np.linalg.norm(x=loss_gradient, ord=2)
            expr = r"  $|\langle\frac{\partial L}{\partial x_i}(x,w)\rangle_{p(w|D)}|_2$"

        print(f"vmin={vmin}\tvmax={vmax}\tgrad_norm={grad_norm}")

        axs[col_idx+1].set_title(f"{grad_norm:.3f}", fontsize=16, weight="bold")
        axs[col_idx+1].set_xlabel(f"Samples = {samples}", fontsize=14, weight="bold")

    axs[0].set_title(f"{expr} =", fontsize=18, weight="bold")

    for ax in axs:
        ax.tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')
        ax.set_xticks([], []) 
        ax.set_yticks([], []) 
        ax.set_xticklabels("")
        ax.set_yticklabels("")

    fig.subplots_adjust(top=0.8)
    fig.tight_layout(h_pad=2, w_pad=2, rect=[0,0,0.93,1])
    return fig

def vanishing_gradients_heatmaps(dataset, loss_gradients_list, n_samples_list, filename, relpath,
                                 norm="linfty"):

    transposed_gradients = np.transpose(np.array(loss_gradients_list), axes=(1, 0, 2, 3))
    if transposed_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should contain the number of samples.")

    vanishing_idxs = compute_vanishing_norms_idxs(loss_gradients=transposed_gradients, 
                                                  n_samples_list=n_samples_list, norm=norm)
    test_images = load_dataset(dataset_name=dataset, n_inputs=np.max(vanishing_idxs)+1, 
                               shuffle=False)[2]

    for im_idx in vanishing_idxs:
        
        original_im = test_images[im_idx].squeeze()
        im_gradients = transposed_gradients[im_idx]
        fig = _vanishing_gradient_heatmap(original_im, im_gradients, 
                                         n_samples_list=n_samples_list, norm=norm)

        path=relpath+filename+"/vanishing_gradients_heatmaps/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path+filename+"_vanGrad_"+str(im_idx)+".png")
        plt.close()


def _get_gradients(args, bnn, test_loader, n_samples_list, relpath):

    filename=bnn.name

    loss_gradients_list = []
    for posterior_samples in n_samples_list:

        if args.compute_grads is True:
            loss_grads = loss_gradients(net=bnn, n_samples=posterior_samples, 
                            savedir=filename+"/", data_loader=test_loader, 
                            device=args.device, filename=filename)
        else:
            loss_grads = load_loss_gradients(n_samples=posterior_samples, filename=filename, 
                                             relpath=relpath, savedir=filename+"/")
        loss_gradients_list.append(loss_grads)
    
    return loss_gradients_list


def main(args):

    rel_path=DATA if args.savedir=="DATA" else TESTS

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # === load BNN and data ===

    dataset, model = saved_BNNs["model_"+str(args.model_idx)]

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name=dataset, batch_size=128, 
                     n_inputs=args.n_inputs, shuffle=False)

    bnn = BNN(dataset, *list(model.values()), inp_shape, out_size)
    bnn.load(device=args.device, rel_path=rel_path)

    # === plot loss gradients ===

    if args.stripplot is True:

        n_samples_list = [1,10,50,100]#,500]
        loss_gradients_list = _get_gradients(args, bnn, test_loader, n_samples_list, rel_path)
        stripplot_gradients_components(loss_gradients_list=loss_gradients_list, 
            n_samples_list=n_samples_list, dataset_name=dataset, filename=bnn.name, relpath=rel_path)

    if args.heatmaps is True:
        
        n_samples_list = [1,10,100]
        args.compute_grads=False
        loss_gradients_list = _get_gradients(args, bnn, test_loader, n_samples_list, rel_path)
        vanishing_gradients_heatmaps(dataset=dataset, loss_gradients_list=loss_gradients_list, 
                                     n_samples_list=n_samples_list, filename=bnn.name, relpath=rel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=1000, type=int, help="input points")
    parser.add_argument("--model_idx", default=0, type=int, help="choose idx from saved_BNNs")
    parser.add_argument("--savedir", default='DATA', type=str, 
                        help="choose dir for loading the BNN: DATA, TESTS")  
    parser.add_argument("--compute_grads", default="False", type=eval, 
                        help="If True compute else load")
    parser.add_argument("--stripplot", default="True", type=eval)
    parser.add_argument("--heatmaps", default="True", type=eval)
    parser.add_argument("--device", default='cuda', type=str, help='cpu, cuda')   
    main(args=parser.parse_args())