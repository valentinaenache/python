import matplotlib.pyplot as plt
import seaborn as sb

def biplot(z, zg, y, regiuni, k1=0, k2=1):
    f = plt.figure(figsize=(12, 8))
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot instante si centre in axele discriminante", fontsize=14)
    ax.set_xlabel("z" + str(k1 + 1))
    ax.set_ylabel("z" + str(k2 + 1))
    sb.scatterplot(z[:, k1], z[:, k2], hue=y, hue_order=regiuni, ax=ax)
    sb.scatterplot(zg[:, k1], zg[:, k2], hue=regiuni, marker="s", s=100, legend=False, ax=ax)

def distributie(z, k, y, regiuni):
    f = plt.figure(figsize=(12, 8))
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Distributie in axa discriminanta z" + str(k + 1), fontsize=16)
    for g in regiuni:
        sb.kdeplot(z[y == g, k], shade=True, label=g, ax=ax)
    ax.legend()

def show():
    plt.show()
