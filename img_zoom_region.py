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

    extent = [440, 560, 440, 560]

    # Get image
    img = ParticleImage(
        0.00233195 * unyt.Mpc * 2,
        fov=120 * unyt.Mpc,
        positions=pos * unyt.Mpc,
        smoothing_lengths=np.full(pos.shape[0], 0.00233195) * unyt.Mpc,
        pixel_values=mass,
        centre=np.array([500, 500, 500])
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
        extent=extent,
        norm=cm.LogNorm(vmin=mass[0] * 0.5, vmax=71203564514.88318, clip=True),
        cmap="swift.nineteen_eighty_nine")

    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False,
                   labelbottom=False)

    ax.text(0.975, 0.06, "$t=$%.1f Gyr" % cosmo.age(redshift).value,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='right', fontsize=6, color="w")

    ax.plot([0.05, 0.05 + 1 / 12], [0.05, 0.05], lw=0.75, color='w',
            clip_on=False,
            transform=ax.transAxes)
    
    ax.plot([0.05, 0.05], [0.04, 0.06], lw=0.75, color='w',
            clip_on=False,
            transform=ax.transAxes)
    ax.plot([0.05 + 1 / 12, 0.05 + 1 / 12], [0.04, 0.06], lw=0.75, color='w',
            clip_on=False,
            transform=ax.transAxes)
    
    axis_to_data = ax.transAxes + ax.transData.inverted()
    left = axis_to_data.transform((0.05, 0.075))
    right = axis_to_data.transform((0.15, 0.075))
    dist = right[0] - left[0]

    ax.text(0.1, 0.06, "%d cMpc" % dist,
            transform=ax.transAxes, verticalalignment="bottom",
            horizontalalignment='center', fontsize=5, color="w")
    
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
