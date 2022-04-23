import matplotlib.pyplot as plt
from util.os_util import make_dir_if_not_exists


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
