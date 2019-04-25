import matplotlib.pyplot as plt

def plot_cost_accuracy(bssa, problem_dim):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("cost")
    ax.set_ylabel("iteration")
    ax.plot(bssa.get_cost_history(), 'g')
    ax.grid()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("err, selected features history")
    sf = bssa.get_selected_features_history()[-1]*problem_dim
    acc = (bssa.get_acc_history()[-1])*100
    ax.plot(bssa.get_acc_history(), 'r', label="acc, final {:.3f} %".format(acc))
    ax.plot(bssa.get_selected_features_history(), 'b', label="selcted features {}".format(sf))
    ax.legend()
    ax.grid()

def plot_costs_accuracies(bssa_list, problem_dim):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("cost")
    ax.set_ylabel("iteration")
    for bssa, i in zip(bssa_list, len(bssa_list)):
        ax.plot(bssa.get_cost_history(), label=i)

