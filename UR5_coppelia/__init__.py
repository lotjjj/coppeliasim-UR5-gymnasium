from gymnasium.envs.registration import register

register(
    id="UR5_coppelia/UR5-v0",
    entry_point="UR5_coppelia.envs:UR5Env",
)
