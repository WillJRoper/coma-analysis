import sys
import h5py
import unyt
import numpy as np
from synthesizer.imaging.images import ParticleImage
import matplotlib.colors as cm
import matplotlib.pyplot as plt
import swiftascmaps
from astropy.cosmology import Planck18 as cosmo


def get_image(snap):

    # Get data
    hdf = h5py.File("../low_res/snapshots/COMA_N155_" + snap + ".hdf5", "r")
    redshift = hdf["Header"].attrs["Redshift"]
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
    
    img.img[img.img < 0] = 0

    print(img.img.max())

    # Plot image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(
        img.img,
        extent=(pos[:, 0].min(), pos[:, 0].max(),
                pos[:, 1].min(), pos[:, 1].max()),
        norm=cm.LogNorm(vmin=mass[0] * 0.5, vmax=71203564514.88318, clip=True),
        cmap="swift.nineteen_eighty_nine")

    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False,
                   labelbottom=False)

    ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(redshift).value,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='right', fontsize=1, color="w")

    ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.1, color='w',
            clip_on=False,
            transform=ax.transAxes)
    
    ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.15, color='w',
            clip_on=False,
            transform=ax.transAxes)
    ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.15, color='w',
            clip_on=False,
            transform=ax.transAxes)
    
    axis_to_data = ax.transAxes + ax.transData.inverted()
    left = axis_to_data.transform((0.05, 0.075))
    right = axis_to_data.transform((0.15, 0.075))
    dist = right[0] - left[0]

    ax.text(0.1, 0.055, "%.2f cMpc" % dist,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='center', fontsize=1, color="w")
    
    plt.margins(0, 0)
    
    fig.savefig("plots/COMA_low_res_test_xy_" + snap + ".png", dpi=300,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
      

snaps = [str(i).zfill(4) for i in range(0, 35)]

# Get snapshot
if len(sys.argv) == 1:

    for snap in snaps:
        get_image(snap)

else:
    
    snap = snaps[int(sys.argv[1])]

    get_image(snap)
