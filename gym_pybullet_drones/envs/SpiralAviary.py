import numpy as np
import pybullet as p
from gymnasium import spaces
import random

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class SpiralAviary(BaseRLAviary):
    """Single-agent spiral trajectory tracking environment without curriculum."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool=False,
                 record: bool=False,
                 mode: str = "spiral",
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        
        self.start_noise_std = 0.0 # was 0.05  
        self.episode_reward = 0
        
        # MODIFIED: Reduced spiral parameters for easier tracking
        self.spiral_radius = 0.5
        self.spiral_height = 0.5
        self.spiral_angular_speed = 0.006   
        self.spiral_vertical_speed = 0.0003
        self.spiral_radial_speed = 0.00006


        # Visualization parameters
        self.vis_target_points = []
        self.vis_drone_points = []
        self.target_line_id = None
        self.drone_line_id = None
        self.vis_step_interval = 1  # More frequent visualization (was 4)
        self.target_sphere_ids = []  # For adding target spheres
        self.last_target_viz = None
        self.drone_color = [random.random(), random.random(), random.random()]

        # for trajectory viz
        self.prev_target_point = None
        self.prev_drone_point  = None


        # Tracking performance
        self.tracking_errors = []

        # MODIFIED: Extended time horizon for slower trajectory
        self.EPISODE_LEN_SEC =20  # Longer episode (was 8)
        self.mode= mode
        self.hover_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Fixed hover point
        self.goto_target = np.random.uniform(low=[-1, -1, 0.5], high=[1, 1, 1.5])
        self.last_action = np.zeros(4, dtype=np.float32)

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
                         
        # Update observation space: include relative position only
        obs_sample = self._computeObs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32
        )

    def _computeSpiralTarget(self, t):
        """Compute the desired spiral target at step t."""
        angle = self.spiral_angular_speed * t
        radius = self.spiral_radius + self.spiral_radial_speed * t
        height = self.spiral_height + self.spiral_vertical_speed * t
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        return np.array([x, y, z], dtype=np.float32)

    def _computeTarget(self, t):
        if self.mode == "spiral":
            return self._computeSpiralTarget(t)
        elif self.mode == "hover":
            return self.hover_target
        elif self.mode == "goto":
            if t % 100 == 0:
                self.goto_target = np.random.uniform(low=[-1, -1, 0.5], high=[1, 1, 1.5])
            return self.goto_target

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        target = self._computeTarget(self.step_counter)

        # MODIFIED: Include velocity of target for better prediction
        if self.step_counter > 0:
            prev_target = self._computeTarget(self.step_counter - 1)
            target_velocity = (target - prev_target) * self.PYB_FREQ  # Scale to get velocity
        else:
            target_velocity = np.zeros(3)
        
        # Relative position to target
        ref_pos = target
        ref_att = [0,0,0]  # Assuming target attitude is zero
        sel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]  # Select all state variables
        delta_pos=ref_pos- state[0:3]
        delta_att=ref_att - state[7:10]
        
        # MODIFIED: Include target velocity in observation
        # return np.hstack([state[sel], ref_pos, ref_att, delta_pos, delta_att]).astype(np.float32) # In case of SAC
        return state

    # def _computeReward(self):
    #     """Computes the current reward value.

    #     Returns
    #     -------
    #     float
    #         The reward.

    #     """
    #     state = self._getDroneStateVector(0)
    #     target = self._computeTarget(self.step_counter)
    #     self.tracking_errors.append(np.linalg.norm(target-state[0:3]))
    #     #ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
    #     ###########################################################################
    #     a = 7
    #     e_k = np.sqrt((target[0]-state[0])**2 + (target[1]-state[1])**2 + (target[2]-state[2])**2)
    #     er1 = 1/(a*e_k)
    #     exp = (-0.5) * ((e_k/0.5)**2)
    #     den = np.sqrt(2*np.pi*(0.5**2))
    #     er2 = (a/den)*np.exp(exp)
    #     reward = er1 + er2
    #     ret=max(0,reward)
    #     return ret
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        target = self._computeTarget(self.step_counter)
        
        # -------------------------------------
        # Position error (Euclidean distance)
        pos_error = np.linalg.norm(target - state[0:3])  # x, y, z error
        self.tracking_errors.append(np.linalg.norm(target-state[0:3]))
        pos_reward = np.exp(-pos_error**2)  # Gaussian decay

        # -------------------------------------
        # Attitude stability (roll/pitch)
        roll, pitch, yaw = state[7:10]
        stability_penalty = (roll**2 + pitch**2)  # Ignore yaw for stability
        stability_reward = np.exp(-stability_penalty * 5)  # Higher weight

        # -------------------------------------
        # Smoothness penalty (optional)
        if self.step_counter > 0:
            action_diff = np.linalg.norm(self.last_action - state[16:20])
        else:
            action_diff = 0.0
        smoothness_penalty = np.exp(-action_diff * 0.1)

        # -------------------------------------
        # Combine
        w_pos = 0.6
        w_stab = 0.3
        w_smooth = 0.1

        total_reward = (
            w_pos * pos_reward +
            w_stab * stability_reward +
            w_smooth * smoothness_penalty
        )

        # Save action for next step
        self.last_action = state[16:20]

        # Clip reward to keep SAC stable
        # total_reward = np.tanh(total_reward)  # Normalize to [-1, 1]

        return total_reward



    def _computeTerminated(self):
        # End episode when time horizon is reached
        return (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        
        # MODIFIED: More forgiving termination conditions
        too_far = np.linalg.norm(pos) > (self.spiral_radius + 2.0)  # Was 3.0
        too_tilted = abs(state[7]) > 1.2 or abs(state[8]) > 1.2  # Was 1.0
        
        return too_far or too_tilted

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        target = self._computeTarget(self.step_counter)
        dist = np.linalg.norm(pos - target)
        
        # Calculate average tracking error for this episode
        avg_error = np.mean(self.tracking_errors) if self.tracking_errors else 0
        
        return {
            "episode": {
                "r": self.episode_reward,
                "l": self.step_counter/self.CTRL_FREQ
            },
            "distance": float(dist),
            "target": target,
            "avg_tracking_error": avg_error
        }
        
    def _updateVisualization(self):
        if not self.GUI:
            return

        t = self.step_counter
        if t % self.vis_step_interval != 0:
            return

        # current positions
        target = self._computeTarget(t)
        drone  = self._getDroneStateVector(0)[0:3]

        # draw target line
        if self.prev_target_point is not None:
            p.addUserDebugLine(self.prev_target_point.tolist(),
                               target.tolist(),
                               [1,0,0],      # red
                               lineWidth=1,
                               lifeTime=0)
        # draw drone line
        if self.prev_drone_point is not None:
            p.addUserDebugLine(self.prev_drone_point.tolist(),
                               drone.tolist(),
                               self.drone_color,      # blue
                               lineWidth=1,
                               lifeTime=0)

        self.prev_target_point = target
        self.prev_drone_point  = drone

    def step(self, action):

        obs, reward, terminated, truncated, info = super().step(action)
        self.episode_reward += reward  # Track cumulative reward
        self.step_counter+=1
        
        # Update visualization after step
        self._updateVisualization()

        
        if terminated or truncated:
            self.episode_reward = 0  # Reset for next episode
            # Clean up visualization elements
            if self.GUI and self.last_target_viz is not None:
                p.removeBody(self.last_target_viz)
                self.last_target_viz = None
            
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Resets the environment and returns the initial observation."""
        if options is not None:
            t0    = options['t0']
            noise = options['noise']
            yaw0  = options['yaw']
        else:
            t0 = self.np_random.integers(0, int(0.1*self.EPISODE_LEN_SEC * self.PYB_FREQ)) # Randomize starting point on spiral
            noise = self.np_random.normal(
                loc=0.05, 
                scale=self.start_noise_std, 
                size=3
            ).astype(np.float32)  # choose noise
            yaw0 = self.np_random.uniform(-np.pi / 4, np.pi / 4)  # Randomize yaw

        self.step_counter = t0

        init_target = self._computeTarget(t0)

        init_pos = init_target + noise        
        # place drone exactly at that spiral point (plus tiny noise if you like)
        self.INIT_XYZS = np.array([[init_pos[0],
                                    init_pos[1],
                                    init_pos[2]]], dtype=np.float32)
        # random yaw around ±45°

        self.INIT_RPYS = np.array([[0.0, 0.0, yaw0]], dtype=np.float32)

        # reset diagnostics & viz
        self.episode_reward = 0
        self.tracking_errors = []
        self.prev_target_point = None
        self.prev_drone_point  = None

        obs, info = super().reset(seed=seed, options=options)

        self.step_counter = t0

        # 🎯 Draw first target
        
        if self.GUI:
            target = self._computeTarget(self.step_counter)
            state = self._getDroneStateVector(0)[0:3]
            self.prev_target_point = target
            self.prev_drone_point  = state

        return obs, info
