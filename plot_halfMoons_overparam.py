"""
Plot gradients components towards the overparametrized limit on half moons dataset.
"""

from grid_search_halfMoons import *
from model_bnn import BNN
import matplotlib
import seaborn as sns

ACC_THS=70

#################################
# exp loss gradients components #
#################################


def plot_half_moons(n_points=200):

    x_train, y_train, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=n_points, channels="first") 
    
    labels = onehot_to_labels(y_train)
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=150, facecolor='w', edgecolor='k')
    df = pandas.DataFrame.from_dict({"x":x_train.squeeze()[:,0],
                                     "y":x_train.squeeze()[:,1],
                                     "label":labels[:]})
    g = sns.scatterplot(data=df, x="x", y="y", hue="label", alpha=0.9, ax=ax)
    filename = "halfMoons_"+str(n_points)+".png"
    os.makedirs(os.path.dirname(PLOTS), exist_ok=True)
    plt.savefig(PLOTS + filename)


def build_overparam_scatterplot_dataset(hidden_size, activation, architecture, 
            inference, epochs, lr, n_samples, warmup, n_inputs, posterior_samples, 
            device, test_points, rel_path):

    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name="half_moons", n_inputs=test_points, channels="first") 

    cols = ["hidden_size", "activation", "architecture", "inference", "epochs", "lr", 
            "n_samples", "warmup", "n_inputs", "posterior_samples", "test_acc",
            "x","y","loss_gradients_x","loss_gradients_y"]
    
    df = pandas.DataFrame(columns=cols)

    combinations = list(itertools.product(hidden_size, activation, architecture, inference, 
                                          epochs, lr, n_samples, warmup, n_inputs))

    row_count = 0

    for init in combinations:
        for n_samples in posterior_samples:

            bnn = MoonsBNN(*init, inp_shape, out_size)
            bnn.load(device=device, rel_path=rel_path)
            bnn_dict = {cols[k]:val for k, val in enumerate(init)}

            test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=64)
            test_acc = bnn.evaluate(test_loader=test_loader, device=device, 
                                    n_samples=n_samples)

            loss_grads = load_loss_gradients(n_samples=n_samples, filename=bnn.name, 
                                             savedir=bnn.name+"/", relpath=rel_path)

            loss_gradients_components = loss_grads[:test_points]
            for idx, grad in enumerate(loss_gradients_components):
                x, y = x_test[idx].squeeze()
                bnn_dict.update({"posterior_samples":n_samples, 
                                 "test_acc":test_acc, "x":x,"y":y,
                                 "loss_gradients_x":grad[0], "loss_gradients_y":grad[1]})
                df.loc[row_count] = pandas.Series(bnn_dict)
                row_count += 1

    print("\nSaving:", df.head())
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    df.to_csv(TESTS+"halfMoons_lossGrads_final_"+str(test_points)+".csv", 
              index=False, header=True)
    return df

def overparam_scatterplot(dataset, hidden_size, test_points, inference, orient="v", device="cuda"):
    dataset = dataset[dataset["test_acc"]>ACC_THS]
    dataset = dataset[dataset["hidden_size"].isin(hidden_size)]
    print("\n---scatterplot_gridSearch_samp_vs_hidden---\n", dataset)

    categorical_rows = dataset["hidden_size"]
    nrows = len(np.unique(categorical_rows))

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'size': 10, 'weight' : 'bold'})
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    # cmap = "rocket_r"
    # cmap = plt.get_cmap('flare', 4)
    cmap = plt.get_cmap('rocket_r', 5)
    cmap = [matplotlib.colors.rgb2hex(cmap(i+1)) for i in range(len(np.unique(dataset["n_inputs"])))]

    if orient == "v":
        num_rows, num_cols = (nrows, 1) 
        figsize = (4, 7)

    else:
        num_rows, num_cols = (1, nrows)
        figsize = (10, 2.3)
    
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, dpi=150, 
                           facecolor='w', edgecolor='k')
    vmin, vmax = (dataset["test_acc"].min(), dataset["test_acc"].max())
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

        
    for r, row_val in enumerate(np.unique(categorical_rows)):
        df = dataset[categorical_rows==row_val]

        legend = "full" if r==3 else None
        g = sns.scatterplot(data=df, x="loss_gradients_x", y="loss_gradients_y", alpha=0.7, 
                            hue="n_inputs", size="n_inputs", legend=legend, 
                            ax=ax[r], sizes=(30, 80), palette=cmap)
        ax[r].set_xlabel("")
        ax[r].set_ylabel("")
        xlim=1.1*np.max(np.abs(df["loss_gradients_x"]))
        ylim=1.1*np.max(np.abs(df["loss_gradients_y"]))
        ax[r].set_xlim(-xlim,+xlim)
        ax[r].set_ylim(-ylim,+ylim)

        if orient == "v":
            ax[r].set_ylabel(str(row_val),labelpad=10,fontdict=dict(weight='bold'),rotation=270) 
            ax[r].yaxis.set_label_position("right")
        else:
            ax[r].set_title(str(row_val), fontdict=dict(weight='bold',size=10)) 
            ax[r].xaxis.set_label_position("bottom")
            ax[r].set_xlabel(r"$\langle \frac{\partial L}{\partial x_1}(x,w)\rangle_{p(w|D)}$", 
                             labelpad=3, fontsize=11)

    ax[0].set_ylabel(r"$\langle \frac{\partial L}{\partial x_2}(x,w)\rangle_{p(w|D)}$",
                     labelpad=3, fontsize=11)

    if orient == "h":
        legend = g.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=1, title="")
        legend.texts[0].set_text("training\ninputs")

    plt.tight_layout()
    filename = "halfMoons_final_hmc_"+str(test_points)+".png"
    os.makedirs(os.path.dirname(PLOTS), exist_ok=True)
    plt.savefig(PLOTS + filename)


def main(args):


    # === settings ===

    inference = ["hmc"]
    n_samples = [50]
    warmup = [100, 200, 500]
    n_inputs = [5000, 10000, 15000]
    epochs = [None]
    lr = [None]
    hidden_size = [32, 128, 256, 512]
    activation = ["leaky"]
    architecture = ["fc2"]
    posterior_samples =  [10,20,50]#[250]

    # === plot ===

    rel_path=DATA if args.savedir=="DATA" else TESTS

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # plot_half_moons(args.test_points)

    # build_overparam_scatterplot_dataset(hidden_size, activation, architecture, inference, 
    #         epochs, lr, n_samples, warmup, n_inputs, posterior_samples, 
    #         device=args.device, test_points=args.test_points, rel_path=rel_path)
    dataset = pandas.read_csv(rel_path+"halfMoons_lossGrads_final_"+str(args.test_points)+".csv")
    overparam_scatterplot(dataset, device=args.device, test_points=args.test_points, 
                        inference=inference, hidden_size=hidden_size, orient="h")

if __name__ == "__main__":
    # assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="Toy example on half moons")
    parser.add_argument("--test_points", default=100, type=int, help="n. of test points")
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
    parser.add_argument("--savedir", default='DATA', type=str, 
                        help="choose dir for loading BNN and gradients: DATA, TESTS")  
    main(args=parser.parse_args())