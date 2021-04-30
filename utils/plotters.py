import os

import constants
import matplotlib.pyplot as plt


def plot_losses(losses, save_folder: str):
    fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel(constants.STEP)
    plt.ylabel(constants.META_LOSS)
    fig.savefig(os.path.join(save_folder, constants.LOSS_PLOT), dpi=100)
