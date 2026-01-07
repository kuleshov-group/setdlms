from src.noise_schedule.noise_schedules import LinearNoise, EaseOutPowerNoise, StaggeredNoise

L = 1024
block_size = 16
num_blocks = L // block_size
max_block_size = L
scale = 1.02

# linear_noise = LinearNoise(block_size=block_size,
#     length=L,
#     plot_schedule=True)

block_diffusion_var = (num_blocks - 1) / (6* num_blocks)

# staggered_noise = StaggeredNoise(
#     scale=scale,
#     length=L,
#     block_size=L,
#     plot_schedule=True,)

ease_out_power_noise = EaseOutPowerNoise(
    block_size=L,
    desired_block_size=block_size,
    max_block_size=max_block_size,
    length=L,
    # b=1,
    plot_schedule=True,
    int_min=0.1)