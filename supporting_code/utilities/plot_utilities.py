import matplotlib.pyplot as plt


def plot_bar(x_axis, y_axis):
    fig = plt.figure()

    ax = fig.add_axes([0,0,1,1])

    ax.bar(x_axis, y_axis)

    plt.show()
