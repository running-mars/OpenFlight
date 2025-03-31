from gym.envs.registration import register
from importlib_metadata import entry_points

register(id="jsbsim-uni-uav-sample-v1", entry_point="airgym.envs:JSBSim3DUniUAVEnv",)       # 3D, velocity-level

register(id="jsbsim-uni-uav-sample-v2", entry_point="airgym.envs:JSBSim3DLLCUniUAVEnv",)    # 3D, low-level