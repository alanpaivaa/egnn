import matplotlib.pyplot as plt
import torch
from util.os_util import make_dir_if_not_exists


def make_node_embeddings_mean_std_directories(stats_dir, batch_idx, trajectory_idx):
    embeddings_dir = '%s/node_embeddings_mean_std' % stats_dir
    make_dir_if_not_exists(embeddings_dir)

    batch_location_dir = '%s/batch_%d' % (embeddings_dir, batch_idx)
    make_dir_if_not_exists(batch_location_dir)

    trajectory_location_dir = '%s/trajectory_%d' % (embeddings_dir, trajectory_idx)
    make_dir_if_not_exists(trajectory_location_dir)

    return batch_location_dir, trajectory_location_dir


def save_epoch_node_embeddings_mean_std(embeddings, epoch, stats_dir, batch_idx, trajectory_idx):
    batch_dir, trajectory_dir = make_node_embeddings_mean_std_directories(
        stats_dir=stats_dir,
        batch_idx=batch_idx,
        trajectory_idx=trajectory_idx
    )

    # Save batch
    embedding = embeddings[batch_idx]
    save_node_embedding_mean_std(
        embedding=embedding,
        filename='%s/epoch_%d' % (batch_dir, epoch),
        title="batch=%d epoch=%d" % (batch_idx, epoch)
    )

    # Save trajectory
    if trajectory_idx < int(embedding.shape[0] / 5):
        embedding = embedding[trajectory_idx * 5:(trajectory_idx + 1) * 5]
    else:
        embedding = embedding[-5:]
    save_node_embedding_mean_std(
        embedding=embedding,
        filename='%s/epoch_%d' % (trajectory_dir, epoch),
        title="trajectory=%d batch=%d epoch=%d" % (trajectory_idx, batch_idx, epoch)
    )


def save_node_embedding_mean_std(embedding, filename, title, rows=2, cols=1, bar_width=0.4):
    mean = torch.mean(embedding, axis=0)
    std = torch.std(embedding, axis=0) * (mean / torch.abs(mean))
    x = torch.arange(std.shape[0])
    slice_size = int(len(x) / (rows * cols))

    subs = list()
    for i in range(0, mean.shape[0], slice_size):
        sub_x = x[i:(i + slice_size)]
        sub_mean = mean[i:(i + slice_size)]
        sub_std = std[i:(i + slice_size)]
        subs.append((sub_x, sub_mean, sub_std))

    plt.close('all')
    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))
    fig.suptitle(title)

    for i in range(rows):
        for j in range(cols):
            sub_x, sub_mean, sub_std = subs.pop(0)
            axs[i].bar(sub_x, sub_mean, label='Mean', width=bar_width, color='mediumseagreen')
            axs[i].bar(sub_x + bar_width, sub_std, label='Standard Deviation', width=bar_width, color='salmon')
            axs[i].set_xticks(sub_x)
            axs[i].set_title("Embedding [%d:%d]" % (sub_x[0], sub_x[-1]))
            axs[i].legend()

    plt.savefig(filename, dpi=300)


def make_location_directories(stats_dir, batch_idx, trajectory_idx):
    locations_dir = '%s/locations' % stats_dir
    make_dir_if_not_exists(locations_dir)

    batch_location_dir = '%s/batch_%d' % (locations_dir, batch_idx)
    make_dir_if_not_exists(batch_location_dir)

    trajectory_location_dir = '%s/trajectory_%d' % (locations_dir, trajectory_idx)
    make_dir_if_not_exists(trajectory_location_dir)

    return batch_location_dir, trajectory_location_dir


def save_epoch_locations(locations, epoch, batch_idx, trajectory_idx, stats_dir):
    # Create directories if needed
    batch_location_dir, trajectory_location_dir = make_location_directories(
        stats_dir=stats_dir,
        batch_idx=batch_idx,
        trajectory_idx=trajectory_idx
    )

    # Save batch locations
    locations = locations[batch_idx]
    save_locations(
        locations=locations,
        title="batch=%d  epoch=%d" % (batch_idx, epoch),
        filename="%s/epoch_%d" % (batch_location_dir, epoch)
    )

    # Save trajectory locations
    if trajectory_idx * 5 < locations.shape[0]:
        locations = locations[trajectory_idx*5:(trajectory_idx+1)*5]
    else:
        locations = locations[-5:]
    save_locations(
        locations=locations,
        title="batch=%d  trajectory=%d  epoch=%d" % (batch_idx, trajectory_idx, epoch),
        filename="%s/epoch_%d" % (trajectory_location_dir, epoch)
    )


def save_locations(locations, title, filename):
    plt.close('all')

    axis = plt.axes(projection="3d")
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')

    # Draw dataset points
    axis.scatter3D(locations[:, 0], locations[:, 1], locations[:, 2])

    plt.title(title)
    plt.savefig(filename, dpi=300)
