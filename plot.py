from directories import *
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_gradient_components(loss_gradients_list, n_samples_list, dataset_name, filename):

    matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')
    # cmap = sns.cubehelix_palette(n_colors=10, start=0.8, rot=0.1, light=0.9, hue=1.5, as_cmap=True)
    # cmap = sns.color_palette("ch:0.8,r=.1,l=.9")
    # sns.set_palette(cmap)
    sns.set_palette("gist_heat", 5)

    loss_gradients_components = []
    plot_samples = []
    for samples_idx, n_samples in enumerate(n_samples_list):
        print("\n\nsamples = ", n_samples, end="\t")
        avg_loss_gradient = np.array(loss_gradients_list[samples_idx]).flatten()
        loss_gradients_components.extend(avg_loss_gradient)
        plot_samples.extend(np.repeat(n_samples, len(avg_loss_gradient)))
        # print(len(loss_gradients),len(loss_gradients[0]),loss_gradients[0].shape, len(loss_gradients_components))

    df = pd.DataFrame(data={"loss_gradients": loss_gradients_components, "n_samples": plot_samples})
    print(df.head())

    sns.stripplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, ax=ax, 
                  jitter=0.2, alpha=0.4)

    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.set_yscale('log')

    # ax.set_title("MNIST", fontsize=10)
    # ax[1].set_title("Fashion MNIST", fontsize=10)

    fig.text(0.5, 0.01, "Samples involved in the expectations ($w \sim p(w|D)$)", ha='center')
    fig.text(0.03, 0.5, r"Expected Gradients components $\langle\nabla L(x,w)\rangle_{w}$", 
             va='center', rotation='vertical')

    path = TESTS+filename+"/"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + filename + "_gradComponents.png")


# def vanishing_gradients_heatmaps(loss_gradients, n_samples_list, fig_idx):

#     transposed_gradients = np.transpose(np.array(loss_gradients), axes=(1, 0, 2))
#     if transposed_gradients.shape[1] != len(n_samples_list):
#         raise ValueError("Second dimension should contain the number of samples.")

#     # save_to_pickle(data=transposed_gradients[533], relative_path=RESULTS+"plots/",
#     #                filename="single_grad_"+str(fig_idx)+".pkl")
#     # save_to_pickle(data=transposed_gradients[794], relative_path=RESULTS+"plots/",
#     #                filename="single_grad_"+str(fig_idx)+".pkl")

#     vanishing_idxs = compute_vanishing_grads_idxs(transposed_gradients, n_samples_list=n_samples_list)
#     vanishing_loss_gradients = transposed_gradients[vanishing_idxs]

#     for im_idx, im_gradients in enumerate(vanishing_loss_gradients):

#         fig = plot_vanishing_gradients(im_gradients, n_samples_list)
#         dir=RESULTS+"plots/vanishing_gradients/"
#         os.makedirs(os.path.dirname(dir), exist_ok=True)
#         fig.savefig(dir+"expLossGradients_vanishingImage_"+str(fig_idx)+"_"+str(im_idx)+".png")

def loss_accuracy(dict, path):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
    ax1.plot(dict['loss'])
    ax1.set_title("loss")
    ax2.plot(dict['accuracy'])
    ax2.set_title("accuracy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
