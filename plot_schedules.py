from src.noise_schedule.noise_schedules import LinearNoise, EaseOutPowerNoise, StaggeredNoise

L = 8
block_size = 2
num_blocks = L // block_size
max_block_size = 8
scale = 4

# linear_noise = LinearNoise(block_size=block_size,
#     length=L,
#     plot_schedule=True)

# staggered_noise = StaggeredNoise(
#     scale=scale,
#     length=L,
#     block_size=L,
#     plot_schedule=True)

ease_out_power_noise = EaseOutPowerNoise(
    block_size=L,
    desired_block_size=block_size,
    max_block_size=max_block_size,
    length=L,
    plot_schedule=True)