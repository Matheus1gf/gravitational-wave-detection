import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

my_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

sns.set(style="whitegrid", font_scale=1.2, palette=sns.color_palette("deep"))

plt.rcParams["figure.figsize"] = (10, 6)

class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def offset_png(x, y, path, ax, zoom=0.1, offset=10):
        img = plt.imread(path)
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False, xybox=(offset, offset), boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    @staticmethod
    def plot_loss_graph(train_losses, valid_losses, epoch, fold):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
        fig.suptitle(f"Fold {fold} | Epoch {epoch}", fontsize=12, y=1.05)
        axes = [ax1, ax2]
        data = [train_losses, valid_losses]
        sns.lineplot(y=train_losses, x=range(len(train_losses)), lw=2.3, ls=":", color=my_colors[3], ax=ax1)
        sns.lineplot(y=valid_losses, x=range(len(valid_losses)), lw=2.3, ls="-", color=my_colors[5], ax=ax2)

        for ax, t, d in zip(axes, ["Treinamento", "Validação"], data):
            ax.set_title(f"{t} - Evolução", size=12, weight='bold')
            ax.set_xlabel("Época", weight='bold', size=9)
            ax.set_ylabel("Perda", weight='bold', size=9)
            ax.tick_params(labelsize=9)
            
        plt.show()