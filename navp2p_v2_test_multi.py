"""
Description:
    - Low-level control for Fixed-wing UAVs towards minimum-time flight
    - Validate the performance with multiple targets

Authors:
    - Running-Mars
"""

import numpy as np
import pandas as pd
import time

from lib.jsbsimlib import JSBSimIO, X8Autopilot
from lib.jsbsim_aircraft import Aircraft, x8
from lib import jsbsim_properties as prp

from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C


class SimLoop:
    def __init__(self,
                 sim_time: float = 1000.,
                 airspeed: float = 30.0,
                 sim_frequency_hz: float = 30.0,
                 aircraft: Aircraft = x8,
                 init_conditions: bool = None,
                 debug_level: int = 0):
        self.sim_time = sim_time
        self.airspeed = airspeed
        self.sim_frequency_hz = sim_frequency_hz
        self.aircraft = aircraft

        self.simIO: JSBSimIO = JSBSimIO(sim_frequency_hz, aircraft, init_conditions, debug_level)
        self.ap: X8Autopilot = X8Autopilot(self.simIO)

        self.state = {
            "goal": np.zeros(3),
            "vel": np.zeros(2),
            "attitude": np.zeros(3),
            "ang_vel": np.zeros(3),
            "collision": False,
        }
        self.goal_num = 64
        self.goal_position_all = np.zeros((self.goal_num, 3))

        for index in range(self.goal_num):
            distance_horiz = 500.
            angle_horiz = 2 * np.pi / 64 * 0.5 + index % 64 * 2 * np.pi / 64 + np.pi
            self.goal_position_all[index, 0] = distance_horiz * np.cos(angle_horiz)
            self.goal_position_all[index, 1] = distance_horiz * np.sin(angle_horiz)
            self.goal_position_all[index, 2] = 20. + (180. - 20.) * index / self.goal_num

        self.reach_radius = 30.                     # when the distance to the goal is less than reach_radius, say arrived!
        self.max_flight_time = 50.

        self.init_conditions = {prp.pitch_rad: 0.,
                                prp.roll_rad: 0.,
                                prp.heading_rad: 0.,
                                prp.lat_geod_deg: 0.,
                                prp.lng_geoc_deg: 0.,
                                prp.altitude_sl_ft: 100. * 3.28,
                                prp.u_fps: 52.48,       # 16.0 m/s
                                prp.v_fps: 0.,
                                prp.w_fps: 0.,
                                prp.p_radps: 0.,
                                prp.q_radps: 0.,
                                prp.r_radps: 0.}

        self.episode_init_time = None
        self.data_path = None

    def _setup_flight(self, index):
        self.simIO.reinitialise(self.init_conditions)
        self.episode_init_time = time.strftime("%Y%m%d-%H%M%S")
        self.data_path = './flight_data_' + str(index).zfill(3) + '.csv'
        df = pd.DataFrame(columns=['sim_time(s)',
                                   'latitude(m)', 'longitude(m)', 'altitude(m)',
                                   'pitch(rad)', 'roll(rad)', 'yaw(rad)',
                                   'velocity_north(m/s)', 'velocity_east(m/s)', 'velocity_down(m/s)',
                                   'velocity_body_front(m/s)', 'velocity_body_right(m/s)', 'velocity_body_down(m/s)',
                                   'roll_rate(rad/s)', 'pitch_rate(rad/s)', 'yaw_rate(rad/s)',
                                   'airspeed(m/s)', 'altitude_rate(m/s)'])
        df.to_csv(path_or_buf=self.data_path, mode='w', index=False)

    def navigate(self, radius=300., altitude=100.):
        for index in range(self.goal_position_all.shape[0]):
            self._setup_flight(index=index)
            self.goal_position = self.goal_position_all[index, :]

            while True:
                sim_time_s = self.simIO.get_time()
                if sim_time_s > self.max_flight_time:
                    break
                position = self.simIO.get_local_position_m()  # return [latitude, longitude, altitude] meter
                orientation = self.simIO.get_local_orientation()  # return [pitch, roll, yaw] rad
                orientation[2] = self.simIO.clip2pi(orientation[2])  # from [0, 2*pi] -> [-pi, pi] rad
                v_ned_mps = self.simIO.get_velocity_mps()  # [v_n, v_e, v_d] meter/s
                v_body_mps = self.simIO.get_velocity_body_mps()
                v_ang_radps = self.simIO.get_angular_velocity_radps()  # roll pitch yaw rate in [-2 * pi, 2 * pi]
                airspeed = self.simIO.get_airspeed_mps()
                altitude_rate = self.simIO.get_altitude_rate_mps()

                goal_yaw = np.arctan2(self.goal_position[1] - position[1], self.goal_position[0] - position[0])
                goal_horiz = np.sqrt(
                    (self.goal_position[1] - position[1]) ** 2 + (self.goal_position[0] - position[0]) ** 2)
                goal_vert = self.goal_position[2] - position[2]

                self.state["goal"][0] = self.simIO.angle_minus(minuend=goal_yaw, subtrahend=orientation[2]) / (
                            2 * np.pi)
                self.state["goal"][1] = np.tanh(goal_horiz * 0.01)  # normalize (especially when d < 200m)
                self.state["goal"][2] = np.tanh(goal_vert * 0.02)  # normalize (especially when d < 100m)

                self.state["vel"][0] = np.tanh((airspeed - 20.) * 0.05)
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

                action, _ = model.predict(obs, deterministic=True)

                # control with Api
                self.simIO[prp.aileron_cmd] = action[0]
                self.simIO[prp.elevator_cmd] = action[1]
                self.simIO[prp.throttle_cmd] = action[2] * 0.5 + 0.5

                # run the Flight Dynamics Model
                self.simIO.run_fdm()

                # flight data recording
                flight_data = {'sim_time(s)': sim_time_s,
                               'latitude(m)': [position[0]],
                               'longitude(m)': [position[1]],
                               'altitude(m)': [position[2]],
                               'pitch(rad)': [orientation[0]],
                               'roll(rad)': [orientation[1]],
                               'yaw(rad)': [orientation[2]],
                               'velocity_north(m/s)': [v_ned_mps[0]],
                               'velocity_east(m/s)': [v_ned_mps[1]],
                               'velocity_down(m/s)': [v_ned_mps[2]],
                               'velocity_body_front(m/s)': [v_body_mps[0]],
                               'velocity_body_right(m/s)': [v_body_mps[1]],
                               'velocity_body_down(m/s)': [v_body_mps[2]],
                               'roll_rate(rad/s)': [v_ang_radps[0]],
                               'pitch_rate(rad/s)': [v_ang_radps[1]],
                               'yaw_rate(rad/s)': [v_ang_radps[2]],
                               'airspeed(m/s)': airspeed,
                               'altitude_rate(m/s)': altitude_rate}
                df = pd.DataFrame(flight_data)
                df.to_csv(path_or_buf=self.data_path, mode='a', index=False, header=False)

                goal_distance = np.sqrt((self.simIO.get_local_position_m()[0] - self.goal_position[0]) ** 2 +
                                        (self.simIO.get_local_position_m()[1] - self.goal_position[1]) ** 2 +
                                        (self.simIO.get_local_position_m()[2] - self.goal_position[2]) ** 2)

                if goal_distance <= self.reach_radius:
                    print("Arrived!")
                    break

            print("* Flight time : %f" % (self.simIO.get_time()))


if __name__ == '__main__':
    model_path = "./nav_p2p_policy_20250304-163217.zip"
    model = A2C.load(model_path)

    simLoop = SimLoop()
    simLoop.navigate()
