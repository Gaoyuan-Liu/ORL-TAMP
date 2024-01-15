from gym.envs.registration import register

register(
    id='Packing-v0',
    entry_point='env:PackingEnv',
)