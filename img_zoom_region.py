import sys
import h5py
import unyt
import numpy as np
from synthesizer.imaging.images import ParticleImage
import matplotlib.colors as cm
import matplotlib.pyplot as plt
import swiftascmaps


# Get snapshot
if len(sys.argv) == 1:

    snaps = [str(i).zfill(4) for i in range(22, 24)]

    for snap in snaps:

        # Get data
        hdf = h5py.File("../low_res/snapshots/COMA_N155_" + snap + ".hdf5", "r")
        hdf["Header"].attrs["Redshift"]
        pos = hdf["PartType1"]["Coordinates"][:]
        mass = hdf["PartType1"]["Masses"][:] * 10 ** 10
        hdf.close()

        print("Got data...")

        # Get image
        img = ParticleImage(
            0.00233195 * unyt.Mpc * 2,
            fov=np.max((np.max(pos[:, 0]) - np.min(pos[:, 0]),
                        np.max(pos[:, 1]) - np.min(pos[:, 1]),
                        np.max(pos[:, 2]) - np.min(pos[:, 2]))) * unyt.Mpc,
            positions=pos * unyt.Mpc,
            smoothing_lengths=np.full(pos.shape[0], 0.00233195) * unyt.Mpc,
            pixel_values=mass
        )
        img.get_hist_img()

        print("Got image...")

        # Plot image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(
            img.img,
            extent=(pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max()),
            norm=cm.LogNorm(),
            cmap="swift.nineteen_eighty_nine")
        fig.savefig("plots/COMA_low_res_test_xy_" + snap + ".png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

else:
    
    snap = sys.argv[1]

    # Get data
    hdf = h5py.File("../low_res/snapshots/COMA_N155_" + snap + ".hdf5", "r")
    hdf["Header"].attrs["Redshift"]
    pos = hdf["PartType1"]["Coordinates"][:]
    mass = hdf["PartType1"]["Masses"][:] * 10 ** 10
    hdf.close()

    print("Got data...")

    # Get image
    img = ParticleImage(
        0.00233195 * unyt.Mpc * 2,
        fov=np.max((np.max(pos[:, 0]) - np.min(pos[:, 0]),
                    np.max(pos[:, 1]) - np.min(pos[:, 1]),
                    np.max(pos[:, 2]) - np.min(pos[:, 2]))) * unyt.Mpc,
        positions=pos * unyt.Mpc,
        smoothing_lengths=np.full(pos.shape[0], 0.00233195) * unyt.Mpc,
        pixel_values=mass
    )
    img.get_hist_img()

    print("Got image...")

    # Plot image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(
        img.img,
        extent=(pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max()),
        norm=cm.LogNorm(),
        cmap="swift.nineteen_eighty_nine")
    fig.savefig("plots/COMA_low_res_test_xy_" + snap + ".png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
