from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

def custom_plot(image, save_path, box=None, polygons=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if box is not None:
        for bb in box:
            rect = Rectangle((int(bb[0]), int(bb[1])), int(bb[2]), int(bb[3]), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    if polygons is not None:
        patches = []
        for p in polygons:
            for p2 in p.polygons:
                polygon = Polygon(p2.numpy().reshape((-1, 2)), False)
                patches.append(polygon)
        p = PatchCollection(patches, alpha=0.4)
        p.set_linewidth(2.0)
        p.set_edgecolor('r')
        ax.add_collection(p)

    plt.axis('off')  # Optional: Turn off the axis
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save the figure to a file
    plt.close(fig)  # Close the figure
