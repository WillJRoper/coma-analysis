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
    fov = extent[1] - extent[0]

    # Get image
    img = ParticleImage(
        0.00233195 * unyt.Mpc,
        fov=fov * unyt.Mpc,
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

    # Define distance indicator
    dist = 10
    left = 0.05 * fov
    right = left + dist
    xmid = left + (dist / 2)
    ymid = 0.05 * fov
    top = ymid + 1.5
    bottom = ymid - 1.5

    ax.plot([left, right], [ymid, ymid], lw=0.75, color='w',
            clip_on=False)
    
    ax.plot([left, left], [bottom, top], lw=0.75, color='w',
            clip_on=False)
    ax.plot([right, right], [bottom, top], lw=0.75, color='w',
            clip_on=False)

    ax.text(xmid, top, "%d cMpc" % dist, verticalalignment="bottom",
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
