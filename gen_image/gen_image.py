import numpy as np
from matplotlib import pyplot as plt

aa = np.load("out.npy").astype(np.uint8)
bb = np.load("ans.npy").astype(np.uint8)

cc = (np.load("in.npy").transpose((0, 2, 3, 1)) * 255).astype(np.uint8)

wh = 255 * np.ones((32, 4, 3), dtype=np.uint8)


def cvt(im):
    x = np.array(([0, 1, 1], [1, 0, 1], [1, 1, 0]))
    cnts = [np.sum(im == i) for i in range(4)]
    arg = np.argmax(cnts)

    if arg == 0:

        return (
            ((im == 1)[:, :, None] * x[0])
            + ((im == 2)[:, :, None] * x[1])
            + ((im == 3)[:, :, None] * x[2])
        ) * 255
    else:
        return (
            ((im == ((arg + 1) % 4))[:, :, None] * x[0])
            + ((im == ((arg + 2) % 4))[:, :, None] * x[1])
            + ((im == ((arg + 3) % 4))[:, :, None] * x[2])
        ) * 255


H = 16
W = 8
for i in range(H):
    for j in range(W):
        plt.subplot(H, W, i * W + j + 1)
        a = cvt(aa[i * W + j])
        b = cvt(bb[i * W + j])
        c = np.concatenate([cc[i * W + j]] * 3, axis=-1)

        plt.imshow(np.concatenate((c, wh, b, wh, a), axis=1))
        plt.axis("off")
        # print(aa[i * 8 + j].max())

plt.savefig("Graylv1.png", dpi=300)
