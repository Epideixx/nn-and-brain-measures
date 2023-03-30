import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from matplotlib import pyplot as plt

# For now on we end up with the following:
#   - energy_predict_from_memory (shape : (n_layers, n_timesteps, n_voxels))
#   - energy_predict_from_hidden (shape : (n_layers, n_timesteps, n_voxels))
#   - real_energy (shape : (n_timesteps, n_voxels))

def correlation(energy_predict_from_memory, energy_predict_from_hidden, real_energy):
    """
    Compute the corelation between both energy_predict_from_memory and energy_predict_from_hidden for each layer, and the real energy.
    """

    correlation_with_memory = np.zeros((energy_predict_from_memory.shape[0], energy_predict_from_memory.shape[2]))
    correlation_with_hidden = np.zeros((energy_predict_from_hidden.shape[0], energy_predict_from_hidden.shape[2]))

    for layer in range(energy_predict_from_memory.shape[0]):
        print("Correlation with memory and hidden for layer {}".format(layer))
        for vox in tqdm(range(energy_predict_from_memory.shape[2]), total = energy_predict_from_memory.shape[2]):
            correlation_with_memory[layer, vox] = pearsonr(energy_predict_from_memory[layer, :, vox], real_energy[:, vox])[0]
            correlation_with_hidden[layer, vox] = pearsonr(energy_predict_from_hidden[layer, :, vox], real_energy[:, vox])[0]


    return correlation_with_memory, correlation_with_hidden # Shape : (n_layers, n_voxels)


def plot_highest_correlation_per_layer(correlation_with_memory, correlation_with_hidden, energy_predict_from_memory, energy_predict_from_hidden, real_energy, save_plot = "Figures/highest_correlation_per_layer.png"):

    # Plot the highest correlation for each layer of memory
    fig, ax = plt.subplots(nrows=correlation_with_memory.shape[0], ncols = 2, figsize=(15, 9))

    for layer in range(correlation_with_memory.shape[0]):
        highest_corr_arg = np.argmax(correlation_with_memory[layer, :])
        ax[layer, 0].plot(energy_predict_from_memory[layer, :, highest_corr_arg], label = "Pred from Memory")
        ax[layer, 0].plot(real_energy[:, highest_corr_arg], label = "Real")
        ax[layer, 0].set_title("Layer {} from Memory".format(layer))
        ax[layer, 0].set_xlabel("Time")
        ax[layer, 0].set_ylabel("Correlation")
        ax[layer, 0].legend(loc = 'upper right')

    for layer in range(correlation_with_hidden.shape[0]):
        highest_corr_arg = np.argmax(correlation_with_hidden[layer, :])
        ax[layer, 1].plot(energy_predict_from_hidden[layer, :, highest_corr_arg], label = "Pred from Hidden")
        ax[layer, 1].plot(real_energy[:, highest_corr_arg], label = "Real")
        ax[layer, 1].set_title("Layer {} from Hidden".format(layer))
        ax[layer, 1].set_xlabel("Time")
        ax[layer, 1].set_ylabel("Correlation")
        ax[layer, 1].legend(loc = 'upper right')

    fig.suptitle('Highest Correlation per Layer', fontsize=16)

    plt.savefig(save_plot)


if __name__ == '__main__':
    n_layers = 2
    n_timesteps = 100
    n_voxels = 1000
    energy_predict_from_memory = np.random.normal(size=(n_layers, n_timesteps, n_voxels))
    energy_predict_from_hidden = np.random.normal(size=(n_layers, n_timesteps, n_voxels))
    real_energy = np.random.normal(size=(n_timesteps, n_voxels))

    correlation_with_memory, correlation_with_hidden = correlation(energy_predict_from_memory, energy_predict_from_hidden,real_energy)

    print(correlation_with_memory[0])

    plot_highest_correlation_per_layer(correlation_with_memory, correlation_with_hidden, energy_predict_from_memory, energy_predict_from_hidden,real_energy)