import matplotlib.pyplot as plt
import torch
from util.os_util import make_dir_if_not_exists


def save_epoch_node_embeddings_std(std, epoch, stats_dir, rows=2):
    std_dir = '%s/std' % stats_dir
    make_dir_if_not_exists(std_dir)

    plt.close('all')
    fig, axs = plt.subplots(rows, figsize=(20, 10))
    fig.suptitle("epoch=%d" % epoch)

    x = torch.arange(std.shape[0]).view(rows, -1)
    std = std.view(rows, -1)

    for i in range(std.shape[0]):
        axs[i].bar(x[i], std[i], label='Standard Deviation')
        axs[i].set_xticks(x[i])
        axs[i].set_title("Embedding [%d:%d]" % (x[i][0], x[i][-1]))
        axs[i].legend()

    filename = "%s/epoch_%d" % (std_dir, epoch)
    plt.savefig(filename, dpi=300)


def save_epoch_locations(locations, epoch, batch_idx, trajectory_idx, stats_dir):
    locations_dir = '%s/locations' % stats_dir
    make_dir_if_not_exists(locations_dir)

    # Save batch locations
    locations = locations[batch_idx]

    # Save trajectory locations
    if trajectory_idx * 5 < locations.shape[0]:
        locations = locations[trajectory_idx*5:(trajectory_idx+1)*5]
    else:
        locations = locations[-5:]

    plt.close('all')

    axis = plt.axes(projection="3d")
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')

    # Draw dataset points
    axis.scatter3D(locations[:, 0], locations[:, 1], locations[:, 2])

    plt.title("batch=%d  trajectory=%d  epoch=%d" % (batch_idx, trajectory_idx, epoch))
    plt.savefig("%s/epoch_%d" % (locations_dir, epoch), dpi=300)
