"""
Ideas:
    1. What is necessary must be ensured!
    2. Less is more!

Description:
    - The representation in polar coordinates is very suitable for fixed-wing flight missions.

Authors:
    - Running-Mars
"""

import numpy as np
import random
import copy

from lib.jsbsimlib import JSBSimIO
from lib.jsbsim_aircraft import x8
from lib import jsbsim_properties as prp

import gym
from gym import spaces


class JSBSim3DLLCUniUAVEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self):
        self.state = {
            "goal": np.zeros(3),
            "vel": np.zeros(2),
            "attitude": np.zeros(3),
            "ang_vel": np.zeros(3),
            "collision": False,
        }
        self.simIO: JSBSimIO = JSBSimIO(sim_frequency_hz=30, aircraft=x8, init_conditions=None, debug_level=0)
        self.goal_position = [500., 0., 100.]   # meters in x, y, z
        self.goal_init_distance = 500.          # in 3d space
        self.goal_distance = None               # in 3d space
        self.goal_distance_horiz = None
        self.goal_distance_vert = None
        self.goal_relative_heading = None

        self.init_conditions = {prp.pitch_rad: 0.,
                                prp.roll_rad: 0.,
                                prp.heading_rad: 0.,
                                prp.lat_geod_deg: 0.,
                                prp.lng_geoc_deg: 0.,
                                prp.altitude_sl_ft: 100. * 3.28,
                                prp.u_fps: 52.48,
                                prp.v_fps: 0.,
                                prp.w_fps: 0.,
                                prp.p_radps: 0.,
                                prp.q_radps: 0.,
                                prp.r_radps: 0.}
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)
        self.episode_init_time = self.simIO.get_sim_dt()
        self._setup_flight()

    def __del__(self):
        self.simIO.close_fdm()

    def _setup_flight(self):
        # goal position randomization, utilizing spherical coordinate system
        theta = random.uniform(1.41, 1.73)      # make the height in [20., 180.] with radius(d) equals 500 m.
        phi = random.uniform(-np.pi, np.pi)
        distance = random.uniform(self.goal_init_distance - 50., self.goal_init_distance + 50.)
        self.goal_position = [distance * np.sin(theta) * np.cos(phi),
                              distance * np.sin(theta) * np.sin(phi),
                              distance * np.cos(theta) + 100.]      # +- 80m around the initialized h

        self.goal_relative_heading = abs(phi)
        self.goal_distance = distance                               # distance in 3d space
        self.goal_distance_horiz = distance * np.sin(theta)         # distance in the horizontal plane
        self.goal_distance_vert = abs(distance * np.cos(theta))     # distance in the vertical plane

        self.pre_action = [0., 0., 0.]

        self.simIO.reinitialise(self.init_conditions)
        self.episode_init_time = self.simIO.get_time()

    def _get_obs(self):
        position = self.simIO.get_local_position_m()                # return [latitude, longitude, altitude] meter
        orientation = self.simIO.get_local_orientation()            # return [pitch, roll, yaw] rad
        orientation[2] = self.simIO.clip2pi(orientation[2])         # from [0, 2*pi] -> [-pi, pi] rad
        v_ang_radps = self.simIO.get_angular_velocity_radps()       # roll pitch yaw rate in [-2 * pi, 2 * pi]
        v_body_mps = self.simIO.get_velocity_body_mps()
        v_ned_mps = self.simIO.get_velocity_mps()                   # north, east, down
        airspeed = self.simIO.get_airspeed_mps()

        goal_yaw = np.arctan2(self.goal_position[1] - position[1], self.goal_position[0] - position[0])
        goal_horiz = np.sqrt((self.goal_position[1] - position[1]) ** 2 + (self.goal_position[0] - position[0]) ** 2)
        goal_vert = self.goal_position[2] - position[2]
        self.state["goal"][0] = self.simIO.angle_minus(minuend=goal_yaw, subtrahend=orientation[2]) / (2 * np.pi)
        self.state["goal"][1] = np.tanh(goal_horiz * 0.01)          # normalize (especially when d < 200m)
        self.state["goal"][2] = np.tanh(goal_vert * 0.02)           # normalize (especially when d < 100m)

        self.state["vel"][0] = np.tanh((v_body_mps[0] - 20.) * 0.05)
        self.state["vel"][1] = np.tanh(v_ned_mps[2] * 0.1)

        self.state["attitude"][0] = orientation[0] / (2 * np.pi)
        self.state["attitude"][1] = orientation[1] / (2 * np.pi)
        self.state["attitude"][2] = orientation[2] / (2 * np.pi)

        self.state["ang_vel"][0] = v_ang_radps[0] / (4 * np.pi)
        self.state["ang_vel"][1] = v_ang_radps[1] / (4 * np.pi)
        self.state["ang_vel"][2] = v_ang_radps[2] / (4 * np.pi)

        obs = np.zeros(11, )
        obs[0:3] = self.state["goal"]
        obs[3:5] = self.state["vel"]
        obs[5:8] = self.state["attitude"]
        obs[8:11] = self.state["ang_vel"]

        return obs

    def _do_action(self, action):
        self.simIO[prp.aileron_cmd] = action[0]
        self.simIO[prp.elevator_cmd] = action[1]
        self.simIO[prp.throttle_cmd] = action[2] * 0.5 + 0.5

    def _compute_reward(self, action, reach_radius=30., time_limit=40.):
        flight_time = self.simIO.get_time() - self.episode_init_time
        position = self.simIO.get_local_position_m()
        orientation = self.simIO.get_local_orientation()
        orientation[2] = self.simIO.clip2pi(orientation[2])
        v_ang_radps = self.simIO.get_angular_velocity_radps()       # roll pitch yaw rate in [-2 * pi, 2 * pi]
        airspeed = self.simIO.get_airspeed_mps()

        self.pre_goal_distance_horiz = copy.deepcopy(self.goal_distance_horiz)
        self.goal_distance_horiz = np.sqrt((position[0] - self.goal_position[0]) ** 2 +
                                           (position[1] - self.goal_position[1]) ** 2)

        self.pre_goal_distance_vert = copy.deepcopy(self.goal_distance_vert)
        self.goal_distance_vert = abs(self.goal_position[2] - position[2])

        self.goal_distance = np.sqrt(self.goal_distance_horiz ** 2 + self.goal_distance_vert ** 2)

        self.pre_goal_relative_heading = copy.deepcopy(self.goal_relative_heading)
        goal_heading = np.arctan2(self.goal_position[1] - position[1], self.goal_position[0] - position[0])
        self.goal_relative_heading = abs(self.simIO.angle_minus(goal_heading, orientation[2]))

        reward_distance_horiz = (self.pre_goal_distance_horiz - self.goal_distance_horiz) * 0.25
        reward_distance_vert = (self.pre_goal_distance_vert - self.goal_distance_vert)
        reward_heading = (self.pre_goal_relative_heading - self.goal_relative_heading) * 20.

        if self.state["collision"]:
            done = 1
        elif position[2] < 10. or position[2] > 190.:
            done = 1
        elif any(abs(element) > 10. for element in v_ang_radps):
            done = 1
        elif self.goal_distance < reach_radius:
            done = 1
        elif flight_time > time_limit:
            done = 1
        else:
            done = 0

        self.pre_action = copy.deepcopy(action)
        reward = reward_distance_horiz + reward_distance_vert + reward_heading

        return reward, done

    def step(self, action):
        self._do_action(action)
        self.simIO.run_fdm()
        obs = self._get_obs()
        reward, done = self._compute_reward(action)

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def close(self):
        self.simIO.close_fdm()

    def render(self):
        return self._get_obs()
