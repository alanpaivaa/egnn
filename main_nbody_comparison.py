import torch
import matplotlib.pyplot as plt
from util.os_util import files_in_dir


def experiment_props_in_dir(stats_dir):
    files = files_in_dir(stats_dir)
    tensor_files = list(filter(lambda f: '.tensor' in f, files))

    ps = list()
    for tf in tensor_files:
        d = dict()
        path = "%s/%s" % (stats_dir, tf)
        d['path'] = path
        d['tensor'] = torch.load(path)
        tf = tf.split('.tensor')[0]
        props = tf.split('__')
        for prop in props:
            pair = prop.split('_')
            d[pair[0]] = '_'.join(pair[1:])
        ps.append(d)

    return ps


def save_experiment_comparison(properties, filename):
    rows = 2
    plt.close('all')
    fig, axs = plt.subplots(rows, 1, figsize=(20, 10))
    fig.suptitle("Embeddings Standard Deviation Comparison", y=0.92, fontsize=16)

    spacing = 0.1
    total_width = 1 - 2 * spacing

    bar_count = len(properties)
    for i in range(rows):
        for j in range(len(properties)):
            tensor = properties[j]['tensor']
            x = torch.arange(tensor.shape[0]).view(rows, -1)
            tensor = tensor.view(rows, -1)
            axs[i].bar(
                x=x[i] - (total_width / 2) + j * (total_width / bar_count),
                height=tensor[i],
                width=total_width / bar_count,
                label="%s layers" % properties[j]['layers'],
                align='edge'
            )
            axs[i].set_xticks(x[i])
            axs[i].set_xlim(x[i][0] - 0.7, x[i][-1] + 0.7)
        axs[i].legend()

    plt.savefig(filename, bbox_inches='tight')


input_dir = 'n_body_stats/comparison'
experiment_properties = experiment_props_in_dir(input_dir)
output = "%s/comparison_output" % input_dir
save_experiment_comparison(experiment_properties, output)
