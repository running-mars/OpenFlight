"""
Descriptions:
    - Wrap the interface of JSBSim for easy use

Functions:
    - Initialize the simulation
    - set and get values of the simulation
    - run the dynamics model
    - plot figures of trajectories and so on
    - autopilot using pid

Notes:
    - Additional interfaces can be wrapped as required.
"""

import os
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Union
from simple_pid import PID
import jsbsim
from lib import jsbsim_properties as prp
from lib.jsbsim_aircraft import Aircraft, x8


class JSBSimIO:
    def __init__(self,
                 sim_frequency_hz: float = 60,
                 aircraft: Aircraft = x8,
                 init_conditions: Dict[prp.Property, float] = None,
                 debug_level: int = 0
                 ):
        self.encoding = 'utf-8'
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(debug_level)
        self.sim_dt = 1.0 / sim_frequency_hz
        self.aircraft = aircraft
        self.fdm.disable_output()
        self.wall_clock_dt = None

        self.initialise(self.sim_dt, self.aircraft.jsbsim_id, init_conditions)

        self.ft_to_m = 0.3048

    def __getitem__(self, prop: Union[prp.BoundedProperty, prp.Property]) -> float:
        """
        Retrieves specified simulation property.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        :param prop: BoundedProperty, the property to be retrieved
        :return: float
        """
        return self.fdm[prop.name]

    def __setitem__(self, prop: Union[prp.BoundedProperty, prp.Property], value) -> None:
        """
        Sets simulation property to specified value.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        Warning: JSBSim will create new properties if the specified one exists.
        If the property you are setting is read-only in JSBSim the operation
        will silently fail.

        :param prop: BoundedProperty, the property to be retrieved
        :param value: object?, the value to be set
        """
        self.fdm[prop.name] = value

    def load_model(self, model_name: str) -> None:
        """
        Load a JSBSim xml formatted aircraft model into the JSBSim flight dynamic model

        :param model_name: name of aircraft model loaded into JSBSim
        :return: None
        """
        load_success = self.fdm.load_model(model_name)

        if not load_success:
            raise RuntimeError('JSBSim could not find specified model name: ' + model_name)

    def get_aircraft(self) -> Aircraft:
        """
        Gets the Aircraft this sim was initialised with.
        """
        return self.aircraft

    def get_loaded_model_name(self):
        """
        Get the name of the loaded aircraft model from the current JSBSim FDM instance

        :return: JSBSim model name
        """
        name: str = self.fdm.get_model_name().decode(self.encoding)
        if name:
            return name
        else:
            return None

    def initialise(self, dt: float, model_name: str, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Start JSBSim with custom initial conditions

        :param dt: simulation rate [s]
        :param model_name: the aircraft model used
        :param init_conditions: initial simulation conditions
        :return: None
        """
        if init_conditions is not None:
            ic_file = 'minimal_ic.xml'
        else:
            ic_file = 'basic_ic.xml'

        ic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ic_file)
        self.fdm.load_ic(ic_path, useAircraftPath=False)
        self.load_model(model_name)
        self.fdm.set_dt(dt)
        self.set_custom_initial_conditions(init_conditions)

        success = self.fdm.run_ic()
        if not success:
            raise RuntimeError('JSBSim failed to initialise simulation conditions.')

    def set_custom_initial_conditions(self, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Set initial conditions different to what is found in the <name-ic.xml> file used

        :param init_conditions: the initial simulation conditions, defined based on prp JSBSim properties
        :return: None
        """
        if init_conditions is not None:
            for prop, value in init_conditions.items():
                self[prop] = value

    def reinitialise(self, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Restart the simulator with initial conditions

        :param init_conditions: the initial simulation conditions, defined based on prp JSBSim properties,
        by default this is the original initialization file
        :return: None
        """
        self.set_custom_initial_conditions(init_conditions=init_conditions)
        no_output_reset_mode = 0
        self.fdm.reset_to_initial_conditions(no_output_reset_mode)

    def run_fdm(self) -> bool:
        """
        Check if the FDM has terminated and if not advances one time step, slows by wall_clock_dt

        :return: True if FDM can advance
        """
        result = self.fdm.run()
        if self.wall_clock_dt is not None:
            time.sleep(self.wall_clock_dt)
        return result

    def close_fdm(self) -> None:
        """
        Close the JSBSim Flight Dynamic Model (FDM) currently running

        :return: None
        """
        if self.fdm:
            self.fdm = None

    def get_sim_dt(self) -> float:
        """ Get JSBSim simulation timestep [s] """
        sim_dt = self[prp.sim_dt]

        return sim_dt

    def get_time(self) -> float:
        """
        Get the current simulation time

        :return: the simulation time
        """
        sim_time = self[prp.sim_time_s]
        return sim_time

    def set_simulation_time_factor(self, time_factor):
        """
        Specifies a factor, relative to realtime, for simulation to run at.
        It should be noted that the frequency control is not precise.

        The simulation runs at realtime for time_factor = 1. It runs at double
        speed for time_factor=2, and half speed for 0.5.

        :param time_factor: int or float, nonzero, sim speed relative to realtime
            if None, the simulation is run at maximum computational speed
        """
        if time_factor is None:
            self.wall_clock_dt = None
        elif time_factor <= 0:
            raise ValueError('time factor must be positive and non-zero')
        else:
            self.wall_clock_dt = self.sim_dt / time_factor

    def get_local_position_m(self) -> list:
        """
        Get the local absolute position from the simulation start point

        :return: position [lat, long, alt] with [meter, meter, meter]
        """
        lat = 111320 * self[prp.lat_geod_deg]
        long = 40075000 * self[prp.lng_geoc_deg] * math.cos(self[prp.lat_geod_deg] * (math.pi / 180.0)) / 360
        alt = self[prp.altitude_sl_ft] * self.ft_to_m
        position = [lat, long, alt]

        return position

    def get_local_orientation(self):
        """
        Get the orientation of the vehicle

        :return: orientation [pitch, roll, yaw] with rad
        """
        pitch = self[prp.pitch_rad]
        roll = self[prp.roll_rad]
        yaw = self[prp.heading_deg] * (math.pi / 180)
        orientation = [pitch, roll, yaw]

        return orientation

    def get_velocity_mps(self):
        v_north_mps = self[prp.v_north_fps] * self.ft_to_m
        v_east_mps = self[prp.v_east_fps] * self.ft_to_m
        v_down_mps = self[prp.v_down_fps] * self.ft_to_m
        v_ned_mps = [v_north_mps, v_east_mps, v_down_mps]

        return v_ned_mps

    def get_velocity_body_mps(self):
        v_body_x_mps = self[prp.u_fps] * self.ft_to_m
        v_body_y_mps = self[prp.v_fps] * self.ft_to_m
        v_body_z_mps = self[prp.w_fps] * self.ft_to_m
        v_body_mps = [v_body_x_mps, v_body_y_mps, v_body_z_mps]

        return v_body_mps

    def get_airspeed_mps(self):
        airspeed_mps = self[prp.airspeed] * self.ft_to_m

        return airspeed_mps

    def get_altitude_rate_mps(self):
        altitude_rate_mps = self[prp.altitude_rate_fps] * self.ft_to_m

        return altitude_rate_mps


    def get_angular_velocity_radps(self):
        p_radps = self[prp.p_radps]     # roll rate
        q_radps = self[prp.q_radps]     # pitch rate
        r_radps = self[prp.r_radps]     # yaw rate
        angular_velocity = [p_radps, q_radps, r_radps]

        return angular_velocity

    def start_engines(self) -> None:
        """ Sets all engines running. """
        self[prp.all_engine_running] = -1

    def set_throttle_mixture_controls(self, throttle_cmd: float, mixture_cmd: float):
        """
        Sets throttle and mixture settings

        If an aircraft is multi-engine and has multiple throttle_cmd and mixture_cmd
        controls, sets all of them. Currently only supports up to two throttles/mixtures.
        """
        self[prp.throttle_cmd] = throttle_cmd
        self[prp.mixture_cmd] = mixture_cmd

        try:
            self[prp.throttle_1_cmd] = throttle_cmd
            self[prp.mixture_1_cmd] = mixture_cmd
        except KeyError:
            pass  # must be single-control aircraft

    def raise_landing_gear(self):
        """ Raises all aircraft landing gear. """
        self[prp.gear] = 0.0
        self[prp.gear_all_cmd] = 0.0

    @staticmethod
    def clip2pi(angle) -> float:
        while angle > math.pi:
            angle = angle - 2 * math.pi
        while angle <= - math.pi:
            angle = angle + 2 * math.pi

        return angle

    @staticmethod
    def clip_value(value: float, low_limit: float, high_limit: float) -> float:
        if value > high_limit:
            value = high_limit
        elif value < low_limit:
            value = low_limit

        return value

    @staticmethod
    def angle_minus(minuend: float, subtrahend: float) -> float:
        minuend = JSBSimIO.clip2pi(minuend)
        subtrahend = JSBSimIO.clip2pi(subtrahend)

        if minuend - subtrahend > math.pi:
            result = minuend - (subtrahend + 2 * math.pi)
        elif minuend - subtrahend < - math.pi:
            result = (minuend + 2 * math.pi) - subtrahend
        else:
            result = minuend - subtrahend

        return result

    @staticmethod
    def flight_visualization(csv_file, obstacles=None, obstacle_radius=50., xy_limit=None) -> None:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 14
        if obstacles is None:
            obstacles = []
        df = pd.read_csv(csv_file, delimiter=',', header=0)
        flight_data = df.to_numpy()
        fig = plt.figure(figsize=(12, 10))
        gri_spec = GridSpec(nrows=4, ncols=2)
        ax1 = fig.add_subplot(gri_spec[0:2, 0], projection='3d')    # 3D trace in ENU frame
        ax2 = fig.add_subplot(gri_spec[0:2, 1])                     # overlooking 2D trace
        ax3 = fig.add_subplot(gri_spec[2, 0])
        ax4 = fig.add_subplot(gri_spec[2, 1])                       # velocity in ENU frame
        ax5 = fig.add_subplot(gri_spec[3, 0])                       # attitude
        ax6 = fig.add_subplot(gri_spec[3, 1])                       # body rate

        ax1.plot(flight_data[:, 2], flight_data[:, 1], flight_data[:, 3])
        max_range = np.array([flight_data[:, 2].max() - flight_data[:, 2].min(),
                              flight_data[:, 1].max() - flight_data[:, 1].min(),
                              flight_data[:, 3].max() - flight_data[:, 3].min()]).max() / 2.0
        mid_x = (flight_data[:, 2].max() + flight_data[:, 2].min()) * 0.5
        mid_y = (flight_data[:, 1].max() + flight_data[:, 1].min()) * 0.5
        mid_z = (flight_data[:, 3].max() + flight_data[:, 3].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        ax1.set_xlabel('longitude(m)')
        ax1.set_ylabel('latitude(m)')
        ax1.set_zlabel('altitude(m)')
        ax1.set_title('3d trace')

        for obstacle in obstacles:
            circle = plt.Circle(xy=(obstacle[1], obstacle[0]), radius=obstacle_radius, color='grey', fill=True)
            ax2.add_artist(circle)
        if xy_limit is not None:
            ax2.set_ylim(xy_limit[0][0], xy_limit[0][1])        # latitude
            ax2.set_xlim(xy_limit[1][0], xy_limit[1][1])        # longitude
        ax2.plot(flight_data[:, 2], flight_data[:, 1])
        ax2.set_xlabel('longitude(m)')
        ax2.set_ylabel('latitude(m)')
        ax2.set_title('2d trace overlooking')
        ax2.set_aspect('equal')
        ax2.grid(alpha=0.3)

        ax3.plot(flight_data[:, 0], flight_data[:, 1], label='lat(m)')
        ax3.plot(flight_data[:, 0], flight_data[:, 2], label='lng(m)')
        ax3.plot(flight_data[:, 0], flight_data[:, 3], label='alt(m)')
        ax3.set_xlabel('time(s)')
        ax3.set_ylabel('position(m)')
        ax3.grid(alpha=0.3)

        ax4.plot(flight_data[:, 0], flight_data[:, 7], label=r'$v_x$')
        ax4.plot(flight_data[:, 0], flight_data[:, 8], label=r'$v_y$')
        ax4.plot(flight_data[:, 0], flight_data[:, 9], label=r'$v_z$')
        ax4.set_xlabel('time(s)')
        ax4.set_ylabel('velocity-NED(m/s)')
        ax4.grid(alpha=0.3)

        ax5.plot(flight_data[:, 0], flight_data[:, 5], label='roll')
        ax5.plot(flight_data[:, 0], flight_data[:, 4], label='pitch')
        ax5.plot(flight_data[:, 0], flight_data[:, 6], label='yaw')
        ax5.set_ylim(-np.pi - 0.2, np.pi + 0.2)
        ax5.set_xlabel('time(s)')
        ax5.set_ylabel('attitude(rad)')
        ax5.grid(alpha=0.3)

        ax6.plot(flight_data[:, 0], flight_data[:, 13], label='roll rate(rad/s)')
        ax6.plot(flight_data[:, 0], flight_data[:, 14], label='pitch rate(rad/s)')
        ax6.plot(flight_data[:, 0], flight_data[:, 15], label='yaw rate(rad/s)')
        ax6.set_xlabel('time(s)')
        ax6.set_ylabel('angular velocity(rad/s)')
        ax6.set_title('angular velocity')
        ax6.grid(alpha=0.3)

        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax6.legend()
        plt.tight_layout()
        plt.show()


class X8Autopilot:
    def __init__(self, sim):
        self.sim = sim
        self.nav = None
        self.track_bearing = 0
        self.track_bearing_in = 0
        self.track_bearing_out = 0
        self.track_distance = 0
        self.flag = False
        self.track_id = -1
        self.state = 0

    def pitch_hold(self, pitch_comm: float) -> None:
        """
        Maintains a commanded pitch attitude [radians] using a PI controller with a rate component

        :param pitch_comm: commanded pitch attitude [radians]
        :return: None
        """
        error = pitch_comm - self.sim[prp.pitch_rad]
        kp = 1.0
        ki = 0.0
        kd = 0.03
        controller = PID(kp, ki, 0.0)
        output = controller(error)
        # with pitch rate
        rate = self.sim[prp.q_radps]
        rate_controller = PID(kd, 0.0, 0.0)
        rate_output = rate_controller(rate)
        output = output+rate_output
        self.sim[prp.elevator_cmd] = output

    def roll_hold(self, roll_comm: float) -> None:
        """
        Maintains a commanded roll attitude [radians] using a PID controller

        :param roll_comm: commanded roll attitude [radians]
        :return: None
        """
        # Nichols Ziegler tuning Pcr = 0.29, Kcr = 0.0380, PID chosen
        error = roll_comm - self.sim[prp.roll_rad]
        kp = 0.20
        ki = kp*0.0  # tbd, should use rlocus plot and pole-placement
        kd = 0.089
        controller = PID(kp, ki, 0.0)
        output = controller(error)
        rate = self.sim[prp.p_radps]
        rate_controller = PID(kd, 0.0, 0.0)
        rate_output = rate_controller(rate)
        output = -output+rate_output        # GZ.W: why here is a negative symbol?
        self.sim[prp.aileron_cmd] = output

    def heading_hold(self, heading_comm: float) -> None:
        """
        Maintains a commanded heading [degrees] using a PI controller

        :param heading_comm: commanded heading [degrees]
        :return: None
        """
        # Attempted Nichols-Ziegler with Pcr = 0.048, Kcr=1.74, lead to a lot of overshoot
        error = heading_comm - self.sim[prp.heading_deg]
        # Ensure the aircraft always turns the shortest way round
        if error < -180:
            error = error + 360
        if error > 180:
            error = error - 360
        kp = -2.0023 * 0.005
        ki = -0.6382 * 0.005
        heading_controller = PID(kp, ki, 0.0)
        output = heading_controller(error)
        if output < - 30 * (math.pi / 180):
            output = - 30 * (math.pi / 180)
        if output > 30 * (math.pi / 180):
            output = 30 * (math.pi / 180)
        self.roll_hold(output)

    def airspeed_hold_w_throttle(self, airspeed_comm: float) -> None:
        """
        Maintains a commanded airspeed [KTAS] using throttle_cmd and a PI controller

        :param airspeed_comm: commanded airspeed [KTAS]
        :return: None
        """
        # Appears fine with simple proportional controller, light airspeed instability at high speed (100kts)
        # airspeed_comm [knot], prp.airspeed [feet/s], 1 feet/s = 0.5925 knots
        error = airspeed_comm - (self.sim[prp.airspeed] * 0.5925)  # set airspeed in KTAS'
        kp = 1.0
        ki = 0.035
        airspeed_controller = PID(kp, ki, 0.0)
        output = airspeed_controller(-error)
        # Clip throttle command from 0 to +1 can't be allowed to exceed this!
        if output > 1:
            output = 1
        if output < 0:
            output = 0
        self.sim[prp.throttle_cmd] = output

    def altitude_hold(self, altitude_comm) -> None:
        """
        Maintains a demanded altitude [feet] using pitch attitude

        :param altitude_comm: demanded altitude [feet]
        :return: None
        """

        error = altitude_comm - self.sim[prp.altitude_sl_ft]
        kp = 0.03
        ki = 0.6
        altitude_controller = PID(kp, ki, 0)
        output = altitude_controller(-error)
        if output < - 10 * (math.pi / 180):
            output = - 10 * (math.pi / 180)
        if output > 15 * (math.pi / 180):
            output = 15 * (math.pi / 180)
        self.pitch_hold(output)

    def yaw_rate_to_heading(self, yaw_rate_comm: float) -> None:
        """
        control the heading with yaw rate using a PI controller

        :param yaw_rate_comm: commanded yaw rate [rad/s]
        :return: None
        """
        # Attempted Nichols-Ziegler with Pcr = 0.048, Kcr=1.74, lead to a lot of overshoot
        error = yaw_rate_comm * 180
        # Ensure the aircraft always turns the shortest way round
        if error < -180:
            error = error + 360
        if error > 180:
            error = error - 360
        kp = -2.0023 * 0.005
        ki = -0.6382 * 0.005
        heading_controller = PID(kp, ki, 0.0)
        output = heading_controller(error)
        if output < - 30 * (math.pi / 180):
            output = - 30 * (math.pi / 180)
        if output > 30 * (math.pi / 180):
            output = 30 * (math.pi / 180)
        self.roll_hold(output)

    def altitude_hold_with_rate(self, altitude_rate_comm) -> None:
        """
        Control altitude rate [feet/s] using pitch attitude

        :param altitude_rate_comm: demanded altitude rate [feet/s] in the range of [-3, 3] m/s
        :return: None
        """

        error = altitude_rate_comm
        kp = 0.03
        ki = 0.6
        altitude_controller = PID(kp, ki, 0)
        output = altitude_controller(-error)
        if output < - 10 * (math.pi / 180):
            output = - 10 * (math.pi / 180)
        if output > 15 * (math.pi / 180):
            output = 15 * (math.pi / 180)
        self.pitch_hold(output)


if __name__ == '__main__':
    jsbSimIO = JSBSimIO()