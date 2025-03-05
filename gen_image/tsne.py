import seaborn as sns
import colorcet as cc
import numpy as np
from matplotlib import pyplot as plt

X_embedded = np.load("tsne.npy")
targ = np.load("targ_tsne.npy")
# np.save("tsne.npy", X_embedded)
# np.save("targ.npy", targ)
plt.figure(figsize=(15, 15))
palette = sns.color_palette(cc.glasbey, n_colors=20)
sns.scatterplot(
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    hue=targ,
    legend="full",
    palette=palette,
    s=3,
)
plt.legend(ncol=2, bbox_to_anchor=(1, 1))

# plt.show()

plt.xlim((-200, 200))
plt.ylim((-200, 200))
plt.savefig("tsne.png", dpi=500)
