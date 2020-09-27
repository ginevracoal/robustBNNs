"""
"""

from adversarialAttacks import *
from model_ensemble import Ensemble_NN


def build_baseline_attacks_df(args):

    rel_path=DATA if args.savedir=="DATA" else TESTS

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    df = pandas.DataFrame(columns=["attack_method", "epsilon", "test_acc", "adv_acc", 
                                   "softmax_rob", "attack_samples","defence_samples",
                                   "model_type"])
    row_count=0
    epsilon=0.3

    ### attack NN

    dataset, hid, activ, arch, ep, lr = saved_NNs["model_"+str(args.model_idx)].values()

    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name=dataset, n_inputs=args.n_inputs)
    test_loader = DataLoader(dataset=list(zip(x_test, y_test)))

    nn = NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size, 
            hidden_size=hid, activation=activ, architecture=arch, epochs=ep, lr=lr)
    nn.load(device=args.device, rel_path=rel_path)
        
    if args.test:
        nn.evaluate(test_loader=test_loader, device=args.device)

    x_test, y_test = (torch.from_numpy(x_test[:args.n_inputs]), 
                      torch.from_numpy(y_test[:args.n_inputs]))
    nn_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=dataset, 
                      device=args.device, method=args.attack_method, filename=nn.name)

    test_acc, adv_acc, softmax_rob = attack_evaluation(net=nn, x_test=x_test, 
                                    x_attack=nn_attack, y_test=y_test, device=args.device)

    for pointwise_rob in softmax_rob:
        df_dict = {"model_type":"nn", "attack_method":args.attack_method, "epsilon":epsilon, 
                   "test_acc":test_acc, "adv_acc":adv_acc, "softmax_rob":pointwise_rob.item(), 
                   "attack_samples":1,"defence_samples":None}

        df.loc[row_count] = pandas.Series(df_dict)
        row_count += 1

    ### attack BNN

    dataset_name, model = saved_BNNs["model_"+str(args.model_idx)]
    
    bnn = BNN(dataset_name, *list(model.values()), inp_shape, out_size)
    bnn.load(device=args.device, rel_path=rel_path)

    if args.test:
        test_loader = DataLoader(dataset=list(zip(x_test, y_test)))
        bnn.evaluate(test_loader=test_loader, device=args.device, n_samples=10)

    bayesian_attack_samples=[1]
    bayesian_defence_samples=[1,50,100]

    for attack_samples in bayesian_attack_samples:
        bnn_attack = attack(net=bnn, x_test=x_test, y_test=y_test, dataset_name=dataset, 
                          device=args.device, method=args.attack_method, filename=bnn.name, 
                          n_samples=attack_samples)

        for defence_samples in bayesian_defence_samples:
            test_acc, adv_acc, softmax_rob = attack_evaluation(net=bnn, x_test=x_test, 
                x_attack=bnn_attack, y_test=y_test, device=args.device, 
                n_samples=defence_samples)

            for pointwise_rob in softmax_rob:
                df_dict = {"model_type":"bnn", "attack_method":args.attack_method, 
                "epsilon":epsilon, "test_acc":test_acc, "adv_acc":adv_acc, 
                "softmax_rob":pointwise_rob.item(), "attack_samples":attack_samples,
                "defence_samples":defence_samples}

                df.loc[row_count] = pandas.Series(df_dict)
                row_count += 1

    ### attack ensemble NN

    ensemble_size = 100
    n_samples_list = [1, 50, 100]

    dataset, hid, activ, arch, ep, lr = saved_NNs["model_"+str(args.model_idx)].values()

    _, _, x_test, y_test, inp_shape, out_size = \
        load_dataset(dataset_name=dataset, n_inputs=args.n_inputs)
    test_loader = DataLoader(dataset=list(zip(x_test, y_test)))

    nn = Ensemble_NN(dataset_name=dataset, input_shape=inp_shape, output_size=out_size, 
            hidden_size=hid, activation=activ, architecture=arch, epochs=ep, lr=lr,
            ensemble_size=ensemble_size)
    nn.load(device=args.device, rel_path=rel_path)
        
    for n_samples in n_samples_list:

        if args.test:
            nn.evaluate(test_loader=test_loader, device=args.device, n_samples=n_samples)

        x_test, y_test = (torch.from_numpy(x_test[:args.n_inputs]), 
                          torch.from_numpy(y_test[:args.n_inputs]))

        nn_attack = attack(net=nn, x_test=x_test, y_test=y_test, dataset_name=dataset, 
                          device=args.device, method=args.attack_method, filename=nn.name)

        test_acc, adv_acc, softmax_rob = attack_evaluation(net=nn, x_test=x_test, 
                                        x_attack=nn_attack, y_test=y_test, device=args.device)

        for pointwise_rob in softmax_rob:
            df_dict = {"model_type":"ensemble", "attack_method":args.attack_method, "epsilon":epsilon, 
                       "test_acc":test_acc, "adv_acc":adv_acc, "softmax_rob":pointwise_rob.item(), 
                       "attack_samples":1,"defence_samples":None}

            df.loc[row_count] = pandas.Series(df_dict)
            row_count += 1

    _save_baseline_attacks_df(df=df, dataset_name=dataset_name, 
        attack_method=args.attack_method, savedir=bnn.name)

    return df 

def _save_baseline_attacks_df(df, dataset_name, attack_method, savedir):

    print("\nSaving:", df)
    os.makedirs(os.path.dirname(TESTS+savedir+"/"), exist_ok=True)
    df.to_csv(TESTS+savedir+"/"+str(dataset_name)+"_baseline_attacks_"+\
        str(attack_method)+".csv", 
              index = False, header=True)
    return df

def load_baseline_attacks_df(dataset_name, attack_method, savedir):
    df = pandas.read_csv(TESTS+savedir+"/"+str(dataset_name)+"_baseline_attacks_"+\
        str(attack_method)+".csv")
    print(df.head(300))
    return df


def lineplot_baseline_attacks(df, dataset_name, attack_method, savedir, n_inputs):
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), dpi=150, facecolor='w', edgecolor='k')

    plt.suptitle(f"{attack_method} attack on {dataset_name}")
    
    xmin, xmax= (df["defence_samples"].min(),df["defence_samples"].max())

    print(np.unique(df[df["model_type"]=="bnn"]["adv_acc"]))


    for idx, row in df.iterrows(): 
        row["defence_samples"]=xmin
        df = df.append(row, ignore_index=True)
        row["defence_samples"]=xmax
        df = df.append(row, ignore_index=True)

    palette=["black","darkred","darkorange"]
    sns.lineplot(data=df, x="defence_samples", y="adv_acc",  
        hue="model_type", ax=ax[0], palette=palette)
    sns.lineplot(data=df, x="defence_samples", y="softmax_rob", 
        hue="model_type", ax=ax[1], palette=palette)

    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')

    filename = str(dataset_name)+"_baseline_attacks_"+str(attack_method)+"_"+\
                str(n_inputs)+".png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + savedir + "/" + filename)


def main(args):

    dataset_name, model = saved_BNNs["model_"+str(args.model_idx)]
    _,_, _, _, inp_shape, out_size = load_dataset(dataset_name=dataset_name, n_inputs=1)
    bnn = BNN(dataset_name, *list(model.values()), inp_shape, out_size)

    if args.attack:

        df = build_baseline_attacks_df(args)

    else:
        df = load_baseline_attacks_df(dataset_name=dataset_name, 
            attack_method=args.attack_method, savedir=bnn.name)

    lineplot_baseline_attacks(df=df, dataset_name=dataset_name, 
        attack_method=args.attack_method, savedir=bnn.name, n_inputs=args.n_inputs)



if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=1000, type=int, help="inputs to be attacked")
    parser.add_argument("--model_idx", default=0, type=int, help="choose model idx")
    parser.add_argument("--test", default=True, type=eval)
    parser.add_argument("--attack", default=True, type=eval)
    parser.add_argument("--n_samples", default=100, type=int)
    parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
    parser.add_argument("--savedir", default='DATA', type=str, help="DATA, TESTS")  
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")   
    main(args=parser.parse_args())
