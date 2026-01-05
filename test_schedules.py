from src.noise_schedule.noise_schedules import EaseOutPowerNoise, LinearNoise, StaggeredNoise
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def simulate_masking(noise, L, block_size, max_block_size, t_steps=1000, masking_trials=1000):
    max_block_sizes = []
    average_block_sizes = []

    t = torch.linspace(0, 1, t_steps).unsqueeze(-1).repeat(1, L)
    move_chance = noise.total_noise(t)
    loc = noise.loc
    w = noise.b
    variances = []
    max_block_sizes = []
    masking_probs = []
    for trial in tqdm(range(1, t.shape[0]), desc="Simulating masking"):
        move_chance_trial = move_chance[trial]
        active = (t[trial, 0] > loc) & (t[trial, 0] <= loc + w)
        variances.append(move_chance_trial.var().item())
        max_block_sizes_trial = []

        # masking for this t
        tokens_predicted = torch.zeros(L)
        for _ in range(masking_trials):
            mask = torch.rand(L) < move_chance_trial.clamp(0, 1)
            valid = mask & active[0] # active AND masked
            tokens_predicted[valid] += 1

            # average_block_sizes.append(valid.sum())
            idx = valid.nonzero()[:, -1]
            if idx.numel() > 1 and 0 in idx:
                max_block_sizes_trial.append(idx[-1] - idx[0] + 1)

        masking_prob_t = tokens_predicted / masking_trials
        masking_probs.append(masking_prob_t)
        if len(max_block_sizes_trial) > 0:
            max_block_sizes.extend(max_block_sizes_trial)

    import ipdb ; ipdb.set_trace()

    # if the first token is being predicted, how many tokens ahead are also being predicted?
    max_block_sizes = torch.tensor(max_block_sizes)
    counts, frequencies = torch.unique(max_block_sizes, return_counts=True)

    plt.figure()
    plt.bar(counts, frequencies / frequencies.sum())
    plt.xlabel("Max block size")
    plt.ylabel("Frequency")
    plt.title("Max block size distribution")
    plt.savefig(f"max_block_size_distribution_L{L}_block_size{block_size}_max_block_size{max_block_size}.png")
    print(f"max_block_size_distribution_L{L}_block_size{block_size}_max_block_size{max_block_size}.png saved")


def sample_permutation_order(noise, L, num_samples=100):
    t = torch.rand(1, L)
    max_deviations = []
    to_permute = torch.ones(L).unsqueeze(0)
    for _ in tqdm(range(num_samples), desc="Sampling permutation order"):
        perm_indices = noise.sample_permutation_order(t, to_permute)
        max_deviation = (perm_indices - torch.arange(0, noise.block_size)[None, None, :]).abs()
        max_deviations.append(max_deviation.max())
    max_deviations = torch.tensor(max_deviations)
    print("max deviation: ", max_deviations.max().item())
    plt.figure()
    plt.hist(max_deviations, bins=100)
    plt.xlabel("Max deviation")
    plt.ylabel("Frequency")
    plt.title("Max deviation distribution")
    plt.savefig(f"max_deviation_distribution_L{L}_block_size{noise.block_size}.png")
    print(f"max_deviation_distribution_L{L}_block_size{noise.block_size}.png saved")
    

if __name__ == "__main__":
    L = 1024
    block_size = 4
    num_blocks = L // block_size
    max_block_size = L
    scale = num_blocks

    # noise = LinearNoise(block_size=block_size,
    #     length=L,
    #     plot_schedule=True)

    # noise = StaggeredNoise(
    #     scale=scale,
    #     length=L,
    #     block_size=L,
    #     plot_schedule=False)

    noise = EaseOutPowerNoise(
        block_size=L,
        desired_block_size=block_size,
        max_block_size=max_block_size,
        length=L,
        plot_schedule=True,
        int_min=0.15)

    # simulate_masking(noise, L, block_size, max_block_size)
    sample_permutation_order(noise, L)