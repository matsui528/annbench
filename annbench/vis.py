import matplotlib.pyplot as plt


def draw(lines, xlabel, ylabel, title, filename, with_ctrl, width, height):
    """
    Visualize search results and save them as an image
    Args:
        lines (list): search results. list of dict.
        xlabel (str): label of x-axis, usually "recall"
        ylabel (str): label of y-axis, usually "query per sec"
        title (str): title of the result_img
        filename (str): output file name of image
        with_ctrl (bool): show control parameters or not
        width (int): width of the figure
        height (int): height of the figure

    """
    plt.figure(figsize=(width, height))

    for line in lines:
        for key in ["xs", "ys", "label", "ctrls", "ctrl_label"]:
            assert key in line

    for line in lines:
        plt.plot(line["xs"], line["ys"], 'o-', label=line["label"])
        if with_ctrl:
            for x, y, ctrl in zip(line["xs"], line["ys"], line["ctrls"]):
                plt.annotate(s=line["ctrl_label"] + ":" + str(ctrl), xy=(x, y),
                             xytext=(x, y+50))

    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.grid(which="both")
    plt.yscale("log")
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.cla()

