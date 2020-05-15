import sys
sys.path.append(".")
from directories import *
from lossGradients import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def stripplot_gradients_components(loss_gradients_list, n_samples_list, dataset_name, filename):

    matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')
    sns.set_palette("gist_heat", 5)

    loss_gradients_components = []
    plot_samples = []
    for samples_idx, n_samples in enumerate(n_samples_list):
        print("\n\nsamples = ", n_samples, end="\t")

        print(f"\tmean = {loss_gradients_list[samples_idx].mean():.4f}", end="\t") 
        print(f"var = {loss_gradients_list[samples_idx].var():.4f}")

        avg_loss_gradient = np.array(loss_gradients_list[samples_idx]).flatten()
        loss_gradients_components.extend(avg_loss_gradient)
        plot_samples.extend(np.repeat(n_samples, len(avg_loss_gradient)))

    df = pd.DataFrame(data={"loss_gradients": loss_gradients_components, 
                            "n_samples": plot_samples})
    print(df.head())

    sns.stripplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, ax=ax, 
                  jitter=0.2, alpha=0.4)

    ax.set_ylabel("")
    ax.set_xlabel("")

    # ax.set_yscale('log')
    # ax.set_title("", fontsize=10)

    fig.text(0.5, 0.01, "Samples involved in the expectations ($w \sim p(w|D)$)", ha='center')
    fig.text(0.03, 0.5, r"Expected Gradients components $\langle\nabla L(x,w)\rangle_{w}$", 
             va='center', rotation='vertical')

    path = TESTS+filename+"/"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + filename + "_gradComponents.png")

def vanishing_gradient_heatmap(gradients, n_samples_list, norm):

    fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list), figsize=(12, 4))
    fig.tight_layout(h_pad=2, w_pad=2)

    vmin, vmax = (np.min(gradients), np.max(gradients))

    for col_idx, samples in enumerate(n_samples_list):
        loss_gradient = gradients[col_idx]
        sns.heatmap(loss_gradient, cmap="YlGnBu", ax=axs[col_idx], square=True, vmin=vmin, 
                    vmax=vmax,  cbar_kws={'shrink': 0.5})
        axs[col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')

        if norm == "linfty":
            norm = np.max(np.abs(loss_gradient))
            expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_\infty$"

        elif norm == "l2":
            norm = np.linalg.norm(x=loss_gradient, ord=2)
            expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_2$"

        axs[col_idx].set_title(f"{expr} = {norm:.3f}", fontsize=11)
        axs[col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)

    return fig

def vanishing_gradients_heatmaps(loss_gradients_list, n_samples_list, filename, norm="l2"):

    transposed_gradients = np.transpose(np.array(loss_gradients_list), axes=(1, 0, 2, 3))
    if transposed_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should contain the number of samples.")

    vanishing_idxs = compute_vanishing_norms_idxs(loss_gradients=transposed_gradients, 
                                                  n_samples_list=n_samples_list, norm=norm)
    vanishing_loss_gradients = transposed_gradients[vanishing_idxs]

    for im_idx, im_gradients in enumerate(vanishing_loss_gradients):

        fig = vanishing_gradient_heatmap(im_gradients, n_samples_list=n_samples_list, 
                                         norm=norm)
        path=TESTS+filename+"/vanishing_gradients_heatmaps/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path+filename+"_vanGrad_"+str(im_idx)+".png")
        plt.close()


def main(args):

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name=args.dataset, batch_size=128, n_inputs=args.inputs,
                     shuffle=True)

    # === load BNN ===
    hidden_size, activation, architecture, inference, \
            epochs, lr, samples, warmup = saved_bnns[args.dataset]

    bnn = BNN(args.dataset, hidden_size, activation, architecture, inference,
              epochs, lr, samples, warmup, inp_shape, out_size)

    bnn.load(device=args.device, rel_path=DATA)
    filename = bnn.name

    # === load base NN ===
    # dataset, epochs, lr, rel_path = ("mnist", 20, 0.001, TRAINED_MODELS)    
    # nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size)
    # nn.load(epochs=epochs, lr=lr, rel_path=rel_path, device=args.device)

    # # === load reduced BNN ===
    # bnn = redBNN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size, 
    #              inference=args.inference, base_net=nn)
    # hyperparams = bnn.get_hyperparams(args)
    # filename = bnn.get_filename(n_inputs=args.inputs, hyperparams=hyperparams)
    # bnn.load(n_inputs=args.inputs, hyperparams=hyperparams, rel_path=TESTS, device=args.device)
    
    # === compute loss gradients ===
    n_samples_list = [1,10,50]#,100, 500

    loss_gradients_list = []
    for posterior_samples in n_samples_list:
        loss_gradients = load_loss_gradients(n_samples=posterior_samples, filename=filename, 
                                             relpath=TESTS, savedir=filename+"/")
        loss_gradients_list.append(loss_gradients)
    
    stripplot_gradients_components(loss_gradients_list=loss_gradients_list, n_samples_list=n_samples_list,
                             dataset_name=args.dataset, filename=filename)

    # vanishing_gradients_heatmaps(loss_gradients_list=loss_gradients_list, 
    #                              n_samples_list=n_samples_list, filename=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot gradients components")
    parser.add_argument("--inputs", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist, fashion_mnist, cifar")
    parser.add_argument("--inference", default="svi", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--samples", default=30, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--device", default='cpu', type=str, help='cpu, cuda')   
    main(args=parser.parse_args())