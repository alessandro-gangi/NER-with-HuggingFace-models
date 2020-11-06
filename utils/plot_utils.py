from os import path
from matplotlib import pyplot as plt
import numpy as np


def plot_results(eval_result: dict, output_dir: str):
    """
    Plot and save three charts representing evaluation scores
    :param eval_result: dict
        A dictionary containing scores of a model evaluation
    :param output_dir: str
        Path to store charts images
    :return: void
    """
    # First (pie) chart representing support by data labels (ex: 'support of B-A', 'support of I-A', ..)
    plot_slices_labels = [l.split('_')[1] for l in eval_result.keys() if 'support' in l and ' ' not in l]
    plot_data = []
    for slice_label in plot_slices_labels:
        plot_data.append(eval_result['eval_' + slice_label + '_support'])

    fig, ax = plt.subplots()
    ax.pie(plot_data, startangle=90)
    ax.axis('equal')
    fig_filename = path.join(output_dir, 'support_by_labels.png')
    legend_labels = ['%s\n%.0f%% (%s)' % (l, s/sum(plot_data)*100, s) for l, s in zip(plot_slices_labels, plot_data)]
    plt.legend(loc="right", labels=legend_labels)
    plt.title('Support by labels', pad=10.0)
    plt.savefig(fig_filename, transparent=True)
    plt.show()

    # Second (bar) chart representing metrics by data labels (ex: 'precision of B-A', 'recall of I-A', ..)
    plot_bars_labels = plot_slices_labels
    plot_group_labels = ['precision', 'recall', 'f1-score']
    plot_data = dict()
    for bar_label in plot_bars_labels:
        bar_values = []
        for gruop_key in plot_group_labels:
            bar_values.append(eval_result['eval_' + bar_label + '_' + gruop_key])
        plot_data.update({bar_label: bar_values})

    fig, ax = plt.subplots()
    bar_plot(ax, plot_data, labels=plot_group_labels, total_width=.7, single_width=.8)
    fig_filename = path.join(output_dir, 'metrics_by_labels.png')
    plt.title('Metrics by labels')
    plt.savefig(fig_filename, transparent=True)
    plt.show()

    # Third (bar) chart representing metrics by types (ex: 'weighted precision, 'micro recall', ..)
    metric_types = ['micro', 'macro', 'weighted', 'samples']
    plot_group_labels = ['avg_precision', 'avg_recall', 'avg_f1-score']

    plot_data = dict()
    for m_type in metric_types:
        bar_values = []
        for pg_label in plot_group_labels:
            bar_values.append(eval_result['eval_' + m_type + ' ' + pg_label])
        plot_data.update({m_type: bar_values})

    fig, ax = plt.subplots()
    bar_plot(ax, plot_data, labels=plot_group_labels, total_width=.7, single_width=.8)
    fig_filename = path.join(output_dir, 'metrics_by_types.png')
    plt.title('Metrics by types')
    plt.savefig(fig_filename, transparent=True)
    plt.show()


def bar_plot(ax, data, labels=None, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    labels: list
        A list with length equal to len(data) containing the ticks labels

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of groups and bars per group
    n_groups = len(list(data.values())[0])
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if labels:
        ax.set_xticks(np.arange(n_groups + 1))
        ax.set_xticklabels(labels)

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

