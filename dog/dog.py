import os
import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data
import time
import math
import numpy as np
import gym
from os import path
import warnings
import random
from .dog_utils import box, quat_to_YZX, PI

DEBUG = False

class Dog(gym.Env):
	"""The main environment object
    Args:
    	mode: (reset mode) stand / sleep / standup
        action_mode (str): "whole": whole RL control, "partial": RL+BO, "residual": residual RL + BO, 
        custom_dynamics: old: (bool) the environment will use dynamics parameters from pyBullet if False
        				 new: (dict) custom dynamics setting, which will overwrite the default one
        version: different state dimension  (Decrepit, V2: "h_body_arm" state mode)
        control_legs: two additional actions from RL agent to control offset angles for two legs
        state_mode (str): "h_body_arm" / "body_arm_p" / "body_arm_leg_full"
        ini_step_param: to provide information for return from reset
        leg_action_mode (str): "none" / "parallel_offset" / "hips_offset" / "knees_offset" / "hips_knees_offset"
        				 in "whole" action_mode, leg_action_mode=="none" is the only choice meaning the RL agent does not directly output leg actions but 
        				 parameters for parameterised leg control, in which case the action space is 6;
        				 in "residual" action_mode, leg_action_mode=="none" means the residual control only involves arm control, and no action from the 
        				 RL agent is for leg control, in which case the action space is 4;
        param_opt: the initial optimising parameters. it wouldn't change until new parameters list is given to self.step() unless debug tuners are enable. whatever length of list given, 
        		   it will be reshaped to length 8 with padding zeros.
        gait: "rose", "triangle", "sine", or "line". only available when the last parameters in param_opt is not equal to 0
        randomise: it could be a float number from 0 to 1
        leg_bootstrapping: developing...

    """
	def __init__(self, render=False, fix_body=False, real_time=False, immortal=False, version=3, normalised_abduct=False,
		mode="stand", action_mode="residual", action_multiplier=0.4, residual_multiplier=0.2, note="", tuner_enable=False, action_tuner_enable=False,
		A_range = (0.01, 1), B_range = (0.01, 0.1), arm_pd_control=False, fast_error_update=False, state_mode="body_arm_p", leg_action_mode="none", leg_offset_multiplier=0.2, 
		ini_step_param=[1, 0.35], experiment_info_str = "", param_opt=[0.015, 0, 0, 6, 0.1, 0.1, 0.1, 0.1, 0], debug_tuner_enable=False, gait="rose", sub_step_callback=None,
		num_history_observation=0, randomise=0, external_force=0, custom_dynamics={}, custom_robot={}, progressing=False, max_steps=1000, leg_offset_range=[-0.6, 0.6], leg_bootstrapping=False,
		only_randomise_dyn=False):
		super(Dog, self).__init__()

		self.render = render
		self.fix_body = fix_body
		self.real_time = real_time
		self.version = version
		self.normalised_abduct = normalised_abduct
		self.mode = mode
		self.immortal = immortal
		self.action_mode = action_mode
		self.residual_multiplier = residual_multiplier
		self.action_multiplier = action_multiplier
		self.note = note
		self.tuner_enable = tuner_enable
		self.action_tuner_enable = action_tuner_enable
		self.A_range = A_range if not progressing else (0, float("inf"))
		self.B_range = B_range
		self.arm_pd_control = arm_pd_control
		self.fast_error_update = fast_error_update
		self.state_mode = state_mode
		self.leg_action_mode = leg_action_mode
		self.leg_offset_multiplier = leg_offset_multiplier
		self.ini_step_param = ini_step_param
		self.experiment_info_str = experiment_info_str
		self.param_opt = param_opt
		self.debug_tuner_enable = debug_tuner_enable
		self.gait = gait
		self.sub_step_callback = sub_step_callback
		self.num_history_observation = num_history_observation
		self.randomise = randomise
		self.custom_dynamics = custom_dynamics
		assert custom_dynamics is not bool  # custom_dynamics now requires a config dict
		self.custom_robot = custom_robot
		self.max_steps = max_steps
		self.leg_offset_range = leg_offset_range

		self.param_opt = (self.param_opt + [0]*9)[:9] # for compatibility
			  
		self.progressing = progressing  # DEVELOPING FEATURE
		self.external_force = external_force
		self.leg_bootstrapping = leg_bootstrapping
		self.only_randomise_dyn = only_randomise_dyn

		self.progressing_A_multiplier = 0

		self.default_dynamcis = {"mass_body": 6.841, "mass_abduct": 0.550, "thigh_shank_mass": 0.200, "toe_mass": 0.0273, "lateralFriction_shank": 0.351, "lateralFriction_toe": 0.512, 
								 "maxJointVelocity": 20.9440, "contactStiffness": 2000, "contactDamping": 20, "linearDamping": 0.04, "angularDamping": 0.04, "jointDamping": 0.01, "max_force": 18, "rl_force_offset": 0,
								 "w_r_offset": 0, "w_p_offset": 0, "w_y_offset": 0, "v_x_offset": 0, "v_y_offset": 0, "v_z_offset": 0, "latency": 0, "control_step": 0.002}


		self.dynamcis_setting = {**self.default_dynamcis, **self.custom_dynamics}

		if self.mode == "standup":
			assert path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "standup_actions.npy"))
			self.standup_actions = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "standup_actions.npy"))

		# robot parameters for IK/FK calculation
		self.k = 0.61 if not "k" in self.custom_robot else self.custom_robot["k"] #important
		self.gamma = 0.97213 if not "gamma" in self.custom_robot else self.custom_robot["gamma"]
		self.a = 0.395 if not "a" in self.custom_robot else self.custom_robot["a"]
		self.b = 0.215 if not "b" in self.custom_robot else self.custom_robot["b"]
		self.c = 0.215 if not "c" in self.custom_robot else self.custom_robot["c"]
		self.l = 0.184 if not "l" in self.custom_robot else self.custom_robot["l"]

		if render:
			# self.client = p.connect(p.GUI) 
			self._p = bc.BulletClient(connection_mode=pybullet.GUI)
			self.real_time_button = self._p.addUserDebugParameter("real time",1, 0, int(self.real_time))
			self.fix_body_button = self._p.addUserDebugParameter("fix body",1, 0, int(self.fix_body))
			self.reset_button = self._p.addUserDebugParameter("reset",1, 0, 0)
			self.reload_button = self._p.addUserDebugParameter("reload",1, 0, 0)
			self.info_button = self._p.addUserDebugParameter("info",1, 0, 0)
			self.reset_button_value = 0
			self.reload_button_value = 0
			self.info_button_value = 0
			self.rpy_tuner_enable = False
			if self.tuner_enable:
				self.rpy_tuner_enable = True
				self.param_tuners = [self._p.addUserDebugParameter("B",0, 0.1, self.param_opt[0]),
									 self._p.addUserDebugParameter("Kp",0, 0.1, self.param_opt[1]),
									 self._p.addUserDebugParameter("Kd",0, 0.1, self.param_opt[2]),
									 self._p.addUserDebugParameter("A/B",0, 10, self.param_opt[3]),
									 self._p.addUserDebugParameter("Kp(arm)",0, 1, self.param_opt[4]),
									 self._p.addUserDebugParameter("Kd(arm)",0, 1, self.param_opt[5]),
									 self._p.addUserDebugParameter("Kp(arm_yaw)",0, 1, self.param_opt[6]),
									 self._p.addUserDebugParameter("Kd(arm_yaw)",0, 1, self.param_opt[7]),
									 self._p.addUserDebugParameter("Delta",0, 0.5, self.param_opt[8])
									 ]

				self.dynamics_tuners = {"contactStiffness": self._p.addUserDebugParameter("contactStiffness", max(0, 0.5*self.dynamcis_setting["contactStiffness"]), 2*self.dynamcis_setting["contactStiffness"], self.dynamcis_setting["contactStiffness"]),
										"contactDamping": self._p.addUserDebugParameter("contactDamping",max(0, self.dynamcis_setting["contactDamping"]*0.5), self.dynamcis_setting["contactDamping"]*2, self.dynamcis_setting["contactDamping"]),
										"lateralFriction_toe": self._p.addUserDebugParameter("lateralFriction_toe", 0, 2, self.dynamcis_setting["lateralFriction_toe"]),
										"lateralFriction_shank": self._p.addUserDebugParameter("lateralFriction_shank",0, 2, self.dynamcis_setting["lateralFriction_shank"]),
										"linearDamping": self._p.addUserDebugParameter("linearDamping", 0, 0.2, self.dynamcis_setting["linearDamping"]),
										"angularDamping": self._p.addUserDebugParameter("angularDamping",0, 0.2, self.dynamcis_setting["angularDamping"]),
										"jointDamping": self._p.addUserDebugParameter("jointDamping",0, 0.2, self.dynamcis_setting["jointDamping"]),
										"maxJointVelocity": self._p.addUserDebugParameter("maxJointVelocity",0, 40, self.dynamcis_setting["maxJointVelocity"]),
										"max_force":self._p.addUserDebugParameter("max_force",0, 240, self.dynamcis_setting["max_force"]), 
										"control_step":self._p.addUserDebugParameter("control_step",0, 0.1, self.dynamcis_setting["control_step"]),
										"mass_body":self._p.addUserDebugParameter("mass_body",0, 100, self.dynamcis_setting["mass_body"])
										}

			if self.action_tuner_enable:
				self.action_tuners = [self._p.addUserDebugParameter("Action0", -1, 1, 0),
									  self._p.addUserDebugParameter("Action1", -1, 1, 0),
									  self._p.addUserDebugParameter("Action2", -1, 1, 0),
									  self._p.addUserDebugParameter("Action3", -1, 1, 0),
									  self._p.addUserDebugParameter("Action4", -1, 1, 0),
									  self._p.addUserDebugParameter("Action5", -1, 1, 0),
									  self._p.addUserDebugParameter("Action6", -1, 1, 0),
									  self._p.addUserDebugParameter("Action7", -1, 1, 0)]

			if self.rpy_tuner_enable:
				self.rpy_tuners = [self._p.addUserDebugParameter("r", -PI, PI, 0),
								   self._p.addUserDebugParameter("p", -PI, PI, -1.3414458169),
								   self._p.addUserDebugParameter("y", -PI, PI, 0)]

			if self.debug_tuner_enable:
				self.debug_tuners = [self._p.addUserDebugParameter("debug0", -1, 1, 0),
								   self._p.addUserDebugParameter("debug1", -1, 1, 0),
								   self._p.addUserDebugParameter("debug2", -1, 1, 0)]
				self.debug_values = [0, 0, 0]

			# self.text_display_button = self._p.addUserDebugParameter("display note",1, 0, 1)
			# self.text = self._p.addUserDebugText(self.note, [0, 0, 1], [1, 1, 1], 2)
		else:
			# self.client = p.connect(p.DIRECT)  
			self._p = bc.BulletClient()

		# print("PYBULLET CLIENT ", self.client, " STARTED")
		self.render = render
		if render:
			import pyautogui


		self.motor_ids = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
		self.motor_ids_r = [0, 1, 2, 4, 8, 9, 10, 12] # including all abduct joints
		self.motor_ids_l = [5, 6, 13, 14]
		# ini_pos = [0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6]
		self.curr_pos = [0.0, -0.795, 0.617, 0.0, -0.795, 0.617, 0.0, -0.94, -1, 0.0, -0.94, -1]

		if not self.leg_action_mode == "none":
			assert action_mode == "residual"

		if action_mode == "partial":
			self.a_dim = 4
		elif action_mode == "whole" :
			self.a_dim = 6 # two for parameterised legs movement
		elif action_mode == "residual":
			if self.leg_action_mode == "none":
				self.a_dim = 4
			elif self.leg_action_mode == "parameter":
				self.a_dim = 6
			elif self.leg_action_mode == "parallel_offset":
				self.a_dim = 6  # each additional action for each leg
			elif self.leg_action_mode == "hips_offset" or self.leg_action_mode == "knees_offset":
				self.a_dim = 6  # each additional action for each leg
			elif self.leg_action_mode == "hips_knees_offset":
				self.a_dim = 8  # four actions for four leg joints

		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.a_dim,), dtype=np.float32)

		#"h_body_arm" / "body_arm_p" / "body_arm_leg_full"
		if self.state_mode == "h_body_arm" or self.version == 2:
			self.s_dim = 14 * (self.num_history_observation + 1)
		elif self.state_mode == "body_arm":
			self.s_dim = 10 * (self.num_history_observation + 1)
		elif self.state_mode == "body_arm_p":
			self.s_dim = 12 * (self.num_history_observation + 1)
		elif self.state_mode == "body_arm_leg_full":
			self.s_dim = 14 * (self.num_history_observation + 1)
		elif self.state_mode == "body_arm_leg_full_p":
			self.s_dim = 16 * (self.num_history_observation + 1)
		elif self.state_mode == "body_arm_leg_full_i":  # DEVELOPING
			self.s_dim = 15 * (self.num_history_observation + 1) 
		elif self.state_mode == "body_arm_leg_full_il":  # DEVELOPING
			self.s_dim = 16 * (self.num_history_observation + 1) 

		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.s_dim,), dtype=np.float32)

		self.t = 0
		self.sub_t = 0
		self.done = False

		self.build_world()
		

		# self.param_domain = [[0.2, 0.6], [0.02,0.06]]
		# self.param_domain = [[0.3, 0.5], [0.03,0.05]]
		# self.step_param = [0, 0]
		# self.step_param_ini = [0.1, 0.01]
		self.step_param_buffered = [0, 0, 0, 0]
		self.step_param_buffer = []

		
		self.theta2 = -1
		self.theta1 = self._theta1_hat(self.theta2)
		self.x_original, self.y_original = self._FK(self.theta1, self.theta2)
		# print(f"THETA1: {self.theta1}    THETA2: {self.theta2}")
		# self.stand_pos = [0.0, -0.795, 0.617, 0.0, -0.795, 0.617, 0.0, self.theta1, self.theta2, 0.0, self.theta1, self.theta2]
		self.stand_pos = [0.0, -PI/2, self._theta2_prime_hat(-PI/2), 0.0, -PI/2, self._theta2_prime_hat(-PI/2), 0.0, -0.94, -1, 0.0, -0.94, -1]

		self.motor_force_l = self.motor_force_r = 50
		self.r_offset, self.p_offset, self.y_offset, self.w_r_offset, self.w_p_offset, self.w_y_offset = 0, 0, 0, 0, 0, 0
		self.x_offset, self.yy_offset, self.z_offset, self.v_x_offset, self.v_y_offset, self.v_z_offset = 0, 0, 0, 0, 0, 0
		self.num_sub_steps = 10
		self.max_vel = 20.9440

		self.latency = 0
		self.control_step = 0.002

		self.p_error = 0
		self.d_error = 0
		self.p_error_yaw = 0
		self.d_error_yaw = 0

		self.p_error_buffer = [0]
		self.d_error_buffer = [0]
		self.p_error_yaw_buffer = [0]
		self.d_error_yaw_buffer = [0]
		self.error_buffer_size = 100

		self.custom_body_mass = 0
		self.sin_value = 0

		self.timer0 = time.time()
		self.leg_offsets = [0, 0, 0, 0]   # -0.6 ~ 0.6
		self.leg_offsets_old = [0, 0, 0, 0]   # -0.6 ~ 0.6

		self.states = []
		self.observations_buffer = []
		self.obs_buffer_size = 10

		self.delta_x = self.param_opt[8] 
		self.progressing_delta_x = self.delta_x

		self.leg_state_indicator = 0 #developing
		self.leg_state_indicators = [0, 0]

		


	def build_world(self):
		self._p.setGravity(0, 0, -9.794) 
		self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

		self.planeId = self._p.loadURDF("plane.urdf")

		self._p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.6])
		
		if self.fix_body and (self.mode == "stand" or self.mode == "standup"):
			self.startpoint = [0, 0, 0.9]
			self.startOrientation = self._p.getQuaternionFromEuler([0,-1.3414458169,0])
		elif self.mode == "stand":
			self.startpoint = [0, 0, 0.5562]#0.54]
			self.startOrientation = self._p.getQuaternionFromEuler([0,-1.3414458169,0])
		elif self.fix_body and self.mode == "sleep":
			self.startpoint = [0, 0, 0.5]
			self.startOrientation = self._p.getQuaternionFromEuler([0,0,0])
		elif self.mode == "sleep" or self.mode == "standup":
			self.startpoint = [0, 0, 0.07]
			self.startOrientation = self._p.getQuaternionFromEuler([0,0,0])

		if path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/mini_cheetah.urdf")):
			self.urdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/mini_cheetah.urdf")
		elif path.exists("models/mini_cheetah.urdf"):
			self.urdf_dir = "models/mini_cheetah.urdf"
		elif path.exists("../models/mini_cheetah.urdf"):
			self.urdf_dir = "../models/mini_cheetah.urdf"

		self.dogId = self._p.loadURDF(self.urdf_dir, basePosition=self.startpoint, baseOrientation=self.startOrientation ,useFixedBase=self.fix_body)
		self._p.changeVisualShape(self.dogId, -1, rgbaColor=[1, 1, 1, 1])
		numJoints = self._p.getNumJoints(self.dogId)
		for j in range(numJoints):
			self._p.changeVisualShape(self.dogId, j, rgbaColor=[1, 1, 1, 1])

		# table_height = 0.4
		# self._p.loadURDF("cube.urdf", [1, 0, 0.5 - (1 - table_height)])  # , globalScaling=0.5
		

		# theta1 = 0.862
		# theta2 = -1.948
		# theta1_prime = -0.795
		# theta2_prime = 0.617
		# ini_pos = [0.0, -0.8, 1.6]*4
		# test_pos = [0.0, theta1_prime, theta2_prime, 0.0, theta1_prime, theta2_prime, 0.0, theta1, theta2, 0.0, theta1, theta2]

		


	def reset(self, full_state=False, freeze=False, reload_urdf=False, param_opt=None):

		if param_opt is not None:
			self.param_opt = param_opt # allow reset the param_opt when resetting
			self.param_opt = (self.param_opt + [0]*8)[:9] # for compatibility

		if reload_urdf:
			self._p.resetSimulation()
			self.build_world()
			self.stateId = self._p.saveState()
		# self._p.removeBody(self.dogId)
		# self.dogId = self._p.loadURDF(self.urdf_dir, basePosition=self.startpoint, baseOrientation=self.startOrientation ,useFixedBase=self.fix_body)
		# self._p.changeVisualShape(self.dogId, -1, rgbaColor=[1, 1, 1, 1])
		# numJoints = self._p.getNumJoints(self.dogId)
		# for j in range(numJoints):
		# 	self._p.changeVisualShape(self.dogId, j, rgbaColor=[1, 1, 1, 1])
		# self._p.removeState(self.dogId)
		if self.render == False:
			assert self.tuner_enable == False

		if self.tuner_enable:

			for dyn_param in self.dynamics_tuners:
				self.dynamcis_setting[dyn_param] = self._p.readUserDebugParameter(self.dynamics_tuners[dyn_param])
			# linearDamping = self._p.readUserDebugParameter(self.dynamics_tuners["linearDamping"])
			# angularDamping = self._p.readUserDebugParameter(self.dynamics_tuners["angularDamping"])
			# jointDamping = self._p.readUserDebugParameter(self.dynamics_tuners["jointDamping"])
			# self._p.changeDynamics(self.planeId, -1, contactStiffness=self._p.readUserDebugParameter(self.dynamics_tuners["contactStiffness"]), contactDamping=self._p.readUserDebugParameter(self.dynamics_tuners["contactDamping"]),
			# 					   linearDamping=linearDamping, angularDamping=angularDamping, jointDamping=jointDamping)
			# abduct_links = [0, 4, 8, 12]
			# thigh_links = [1, 5, 9, 13]
			# shank_links = [2, 6, 10, 14]
			# toe_links = [3, 7, 11, 15]
			# for link in abduct_links:
			# 	self._p.changeDynamics(self.dogId, link, jointDamping=jointDamping)
			# for link in shank_links:
			# 	self._p.changeDynamics(self.dogId, link, lateralFriction=self._p.readUserDebugParameter(self.dynamics_tuners["lateralFriction_shank"]))
			# for link in toe_links:
			# 	self._p.changeDynamics(self.dogId, link, lateralFriction=self._p.readUserDebugParameter(self.dynamics_tuners["lateralFriction_toe"]))
			# for link in thigh_links:
			# 	self._p.changeDynamics(self.dogId, link, jointDamping=jointDamping)

			# self.max_vel = self.dynamics_tuners["maxJointVelocity"]
			# self.max_force = self.dynamics_tuners["max_force"]
			# self.control_step = self.dynamics_tuners["max_force"]
			# self._p.setPhysicsEngineParameter(fixedTimeStep=self.control_step)  localInertiaDiagnoal localInertiaDiagonal

			# print("[DEBUG A] inertia: ", self._p.getDynamicsInfo(self.dogId, -1)[2])

		if self.randomise:
			half_range = self.randomise / 10
			random_dynamics = {"mass_body": np.random.uniform(self.dynamcis_setting["mass_body"]*(1-half_range), self.dynamcis_setting["mass_body"]*(1+half_range)),
							   # "jointDamping": np.random.uniform(0, 0.04),
							   "jointDamping": np.random.uniform(self.dynamcis_setting["jointDamping"]*(1-half_range), self.dynamcis_setting["jointDamping"]*(1+half_range)),
							   "max_force": np.random.uniform(self.dynamcis_setting["max_force"]*(1-half_range), self.dynamcis_setting["max_force"]*(1+half_range)),
							   "maxJointVelocity": np.random.uniform(self.dynamcis_setting["maxJointVelocity"]*(1-half_range), self.dynamcis_setting["maxJointVelocity"]*(1+half_range)),
							   "lateralFriction_toe": np.random.uniform(self.dynamcis_setting["lateralFriction_toe"]*(1-half_range), self.dynamcis_setting["lateralFriction_toe"]*(1+half_range)),
							   "contactStiffness": np.random.uniform(self.dynamcis_setting["contactStiffness"]*(1-half_range), self.dynamcis_setting["contactStiffness"]*(1+half_range)),  # Note that in some experiments this is not randomised
							   "contactDamping": np.random.uniform(self.dynamcis_setting["contactDamping"]*(1-half_range), self.dynamcis_setting["contactDamping"]*(1+half_range)) # Note that in some experiments this is not randomised
							   # "latency": np.random.uniform(0, 0.06),
							   # "control_step": np.random.uniform(0.001, 0.003)
							   }
			if 'k' in self.custom_robot:
				self.k = np.random.uniform(self.custom_robot["k"]*(1-half_range), self.custom_robot["k"]*(1+half_range))
			else:
				self.k = np.random.uniform(0.61*(1-half_range), 0.61*(1+half_range))
		else:
			random_dynamics = {}

		dynamcis_setting_randomlised = {**self.dynamcis_setting, **random_dynamics}

		# if DEBUG:
		# 	print("SETTING DYNAMICS: ", dynamcis_setting_randomlised)

		self._set_dynamics(**dynamcis_setting_randomlised)



		# for link in [-1, 0, 4, 1, 5, 2, 6]:
		# 	self._p.changeDynamics(self.dogId, link, localInertiaDiagonal=np.array([0.05, 0.08, 0.1])*2000)


		
		if self.fix_body and (self.mode == "stand" or self.mode == "standup"):
			self.startpoint = [0, 0, 0.9]
			self.startOrientation = self._p.getQuaternionFromEuler([0,-1.3414458169,0])
			self._p.changeDynamics(self.dogId, -1, mass=0)
		elif self.mode == "stand":
			self.startpoint = [0, 0, 0.5562]#0.54]
			self.startOrientation = self._p.getQuaternionFromEuler([0,-1.3414458169,0])
		elif self.fix_body and self.mode == "sleep":
			self.startpoint = [0, 0, 0.27]
			self.startOrientation = self._p.getQuaternionFromEuler([0,0.05,0])
		elif self.mode == "sleep" or self.mode == "standup" :
			self.startpoint = [0, 0, 0.07]
			self.startOrientation = self._p.getQuaternionFromEuler([0,0,0])


		if hasattr(self, "stateId"):
			self._p.restoreState(self.stateId)
		else:
			self.stateId = self._p.saveState()

		self._p.resetBasePositionAndOrientation(self.dogId, self.startpoint, self.startOrientation)
		self._p.resetBaseVelocity(self.dogId, [0]*3, [0]*3)

		self.r_offset, self.p_offset, self.y_offset = 0, 0, 0
		self.x_offset, self.yy_offset, self.z_offset = 0, 0, 0

		self.action_test = [1, 1, -1, -1]

		if self.mode == "stand":
			self.ini_pos = [0.0, -PI/2, self._theta2_prime_hat(-PI/2), 0.0, -PI/2, self._theta2_prime_hat(-PI/2), 0.0, self._theta1_hat(-1), -1, 0.0, self._theta1_hat(-1), -1]  
			# self.ini_pos = [0.0, -PI/2, self._theta2_prime_hat(-PI/2), 0.0, -PI/2, self._theta2_prime_hat(-PI/2), 0.0, -0.94, -1, 0.0, -0.94, -1]  # UPDATE: THETA 1 IS CALCULATED BY THETA 2 
		elif self.mode == "sleep" or self.mode == "standup":
			self.ini_pos = [-0.5, -1.1, 2.79, 0.5, -1.1, 2.79, -1, -1.4, 2, 1, -1.4, 2]

		for j, pos in zip(self.motor_ids, self.ini_pos):
			self._p.resetJointState(self.dogId, j, pos, 0)

		_, quat = self._p.getBasePositionAndOrientation(self.dogId)
		_, omega = self._p.getBaseVelocity(self.dogId)
		
		# print("omega pitch after reseting: ", omega[1], "pitch after reseting: ", quat_to_YZX(self._p.getBasePositionAndOrientation(self.dogId)[1])[1])
		if np.sum(np.array(quat) - np.array(self.startOrientation)) > 0.00000001:
			print("[WARN] RESET MAY BE FAILED ...")

		for i in range(random.randint(1,10)):

			for j, pos in zip(self.motor_ids, self.ini_pos):
				self._p.setJointMotorControl2(self.dogId, j, self._p.POSITION_CONTROL, pos, force=self.motor_force_r)
			self._p.stepSimulation() 
			if self.real_time:
				time.sleep(0.001)

		if self.mode == "standup":
			for a in self.standup_actions:
				self.step_jpos(a)
				if self.real_time:
					time.sleep(0.001)

			_, quat = self._p.getBasePositionAndOrientation(self.dogId)
			if not np.sum(np.array(quat) - np.array(self.startOrientation)) < 0.00000001:
				print(f"RESET STEP {i}: ERROR IS {np.sum(np.array(quat) - np.array(self.startOrientation)) }")


		self.t = 0
		self.sub_t = 0
		self.done = False
		if not full_state:
			self.states = []  # IMPORTANT

			if self.state_mode[-2:] == "_p":  # THIS MAY BE PROBLEMATIC IF OBSERVATION HISTORY IS NOT 0
				state = list(self._get_state(pure_state=True)[0]) + self.ini_step_param
			else:

				state = self._get_state()[0]
			if self.version == 3:
				self.pitch_ref = state[1]  # -1.3406621190226633
			assert np.array(np.array(state)).size == self.observation_space.shape[0]

			
		else:
			state = self.get_full_state()

		if freeze:
			while True:
				pass

		original_state = [i for i in state]

		if not self.leg_bootstrapping:
			# if self.leg_bootstrapping==True, do not initialise two leg offsets vectors
			self.leg_offsets = [0]*4   # some experiemnts forgot to initialise this value !!!
			self.leg_offsets_old = [0]*4   # some experiemnts forgot to initialise this value !!!

		self.leg_state_indicator = 0 #developing
		self.leg_state_indicators = [0, 0]

		self.p_error_buffer = [0]
		self.d_error_buffer = [0]
		self.p_error_yaw_buffer = [0]
		self.d_error_yaw_buffer = [0]

		self.observations_buffer = []
		self.observations_buffer.append(state)
		self.observations_buffer = self.observations_buffer[-self.obs_buffer_size:]
		obs_idx = int(self.latency / (self.control_step*10))  # for example, if control_step is 0.002 and latency is 0.06, the agent should observe the observation 3 steps before 
		state = self.observations_buffer[-(obs_idx+1)] if obs_idx+1 < len(self.observations_buffer) else self.observations_buffer[0]
		state = np.array(state)

		if self.latency == 0:
			assert np.all(np.array(original_state) == state)

		if self.progressing:
			self.progressing_A_multiplier = 0
			self.progressing_delta_x = self.param_opt[8]

		return state

	def step(self, action=[], param=None, param_opt=None):

		if DEBUG:
			# action = [0]*6
			assert self.progressing == False
			assert self.latency == 0
			assert self.randomise == False
			assert self.num_history_observation == 0
			assert self.external_force == False

		


		# action: from-1 to 1, incremental position
		# print(action)

		# if (self.t % 100 == 0):
		# 	random.shuffle(self.action_test)
		# print("================= TORSO =================")
		# # self._p.changeDynamics(self.dogId, -1, mass=110, linearDamping=0.1, angularDamping=0.1, jointDamping=0.2)
		# # self._p.changeDynamics(self.dogId, 0, mass=119, linearDamping=0.1, angularDamping=0.1, jointDamping=0.2)
		# self.set_dynamics(mass_body=996, mass_abduct=555)
		# print(self._p.getDynamicsInfo(self.dogId, -1))

		# print("================= ABDUCT =================")
		# print(self._p.getDynamicsInfo(self.dogId, 0))

		# print("================= THIGH =================")
		# print(self._p.getDynamicsInfo(self.dogId, 1))

		# print("================= SHANK =================")
		# print(self._p.getDynamicsInfo(self.dogId, 2))

		# print("================= TOE =================")
		# print(self._p.getDynamicsInfo(self.dogId, 11))


		# print("jointDamping: ", self._p.getJointInfo(self.dogId, self.motor_ids[0])[6])

		# print("[DEBUG C] inertia: ", self._p.getDynamicsInfo(self.dogId, -1)[2])


		assert param == None   # this mode is not supported any more

		if param_opt is not None:
			self.param_opt = param_opt
			self.param_opt = (self.param_opt + [0]*8)[:9] # for compatibility



		if self.render:
			self.real_time = self._p.readUserDebugParameter(self.real_time_button) % 2 != 0  
			self.fix_body = self._p.readUserDebugParameter(self.fix_body_button) % 2 != 0  
			# if not self.fix_body:
			# 	self._p.changeDynamics(self.dogId, -1, mass=self.custom_body_mass)

			if self.tuner_enable:
				# it is not necessary to check self.arm_pd_control. parameters for arms would all be zero when self.arm_pd_control == True
				self.param_opt = [self._p.readUserDebugParameter(tuner) for tuner in self.param_tuners[:9]]
				self.delta_x = self.param_opt[8] # if not self.progressing else self.progressing_delta_x



			# self.progressing = True if self.param_opt[3] == 0 else False  # DEVELOPING FEATURE
			# self.progressing = False # TODO


			reset_button_value = self._p.readUserDebugParameter(self.reset_button)
			if reset_button_value != self.reset_button_value:
				self.reset_button_value = reset_button_value
				s = self.reset()
				return s, 0, False, {"p_error":0, "d_error":0}
			reload_button_value = self._p.readUserDebugParameter(self.reload_button)
			if reload_button_value != self.reload_button_value:
				self.reload_button_value = reload_button_value
				s = self.reset(reload_urdf=True)
				return s, 0, False, {"p_error":0, "d_error":0}
			info_button_value = self._p.readUserDebugParameter(self.info_button)
			if info_button_value != self.info_button_value:
				self.info_button_value = info_button_value
				pyautogui.alert(self.experiment_info_str, self.note)  # always returns "OK"

			if self.action_tuner_enable:
				if self.leg_action_mode == "none":
					action = [self._p.readUserDebugParameter(tunner) for tunner in self.action_tuners[:4]]
				elif self.leg_action_mode == "parallel_offset" or self.leg_action_mode == "hips_offset" or self.leg_action_mode == "knees_offset" or self.leg_action_mode == "parameter":
					action = [self._p.readUserDebugParameter(tunner) for tunner in self.action_tuners[:6]]
				elif self.leg_action_mode == "hips_knees_offset":
					action = [self._p.readUserDebugParameter(tunner) for tunner in self.action_tuners[:8]]


			if self.fix_body and self.rpy_tuner_enable:
				self.startOrientation = self._p.getQuaternionFromEuler([self._p.readUserDebugParameter(tunner) for tunner in self.rpy_tuners])
				self._p.resetBasePositionAndOrientation(self.dogId, self.startpoint, self.startOrientation)

			if self.debug_tuner_enable:
				self.debug_values = [self._p.readUserDebugParameter(tunner) for tunner in self.debug_tuners]

		max_iter = self.num_sub_steps-1

		if action==[]:
			for i in range(max_iter+1):
				self.sub_t += 1
				self._p.stepSimulation() 
				if self.real_time:
					time.sleep(0.001)

			state, x, height = self._get_state()
			reward = height + x + 0.5

			self.t += 1
			if self.t > self.max_steps or height < 0.2 or self.pitch > -0.7: #-0.27:
				self.done = True
			return state, reward, self.done, {"height": height}


		if np.array(action).size == 12:
			for j, pos in zip(self.motor_ids_r, action):
				self._p.setJointMotorControl2(self.dogId, j, self._p.POSITION_CONTROL, pos, force=self.motor_force_r, maxVelocity=self.max_vel)
			for j, pos in zip(self.motor_ids_l, action):
				self._p.setJointMotorControl2(self.dogId, j, self._p.POSITION_CONTROL, pos, force=self.motor_force_l, maxVelocity=self.max_vel)

			self.sub_t += 1
			self.t += 1
			self._p.stepSimulation() 
			if self.real_time:

				time.sleep(0.001)

			return self.get_full_state(), 0, self.done, 0


		joints_state_old = self._p.getJointStates(self.dogId, self.motor_ids)
		joints_pos_old = [joints_state_old[i][0] for i in range(12)]

		action_multiplier = self.action_multiplier
		
		pos_impl = [joints_pos_old[0] + action_multiplier*action[0], joints_pos_old[1] + action_multiplier*action[1], 0,
					joints_pos_old[3] + action_multiplier*action[2], joints_pos_old[4] + action_multiplier*action[3], 0,
					0, 0, 0,
					0, 0, 0]

		
		self.step_param_buffered[2] = - (self.param_opt[4] * self.p_error + self.param_opt[5] * self.d_error)
		self.step_param_buffered[3] = - (self.param_opt[6] * self.p_error_yaw + self.param_opt[7] * self.d_error_yaw)
		# print("D ERROR: ", self.d_error_yaw)
		pos_impl[1] += self.step_param_buffered[2]
		pos_impl[4] += self.step_param_buffered[2]

		# print(f"{pos_impl[0]} += {self.step_param_buffered[3]}")
		# print(f"{pos_impl[1]} += {self.step_param_buffered[2]}")

		if self.normalised_abduct:
			if pos_impl[1] > -PI/2 - 0.174:
				pos_impl[0] = joints_pos_old[0] + (action_multiplier*action[0] + self.step_param_buffered[3])
			else:
				pos_impl[0] = joints_pos_old[0] - (action_multiplier*action[0] + self.step_param_buffered[3])

			if pos_impl[4] > -PI/2 - 0.174:
				pos_impl[3] = joints_pos_old[3] + (action_multiplier*action[2] + self.step_param_buffered[3])
			else:
				pos_impl[3] = joints_pos_old[3] - (action_multiplier*action[2] + self.step_param_buffered[3])

		if pos_impl[1] > -PI/2 - 0.174:
			ad_right_limit = (-PI/3, 0)
		else:
			ad_right_limit = (0,PI/3)

		if pos_impl[4] > -PI/2 - 0.174:
			ad_left_limit = (0,PI/3)
		else:
			ad_left_limit = (-PI/3, 0)

		pos_impl, reach_limit = box(pos_impl, [ad_right_limit, (-PI, 0), (-PI/2, PI/2)]
											+ [ad_left_limit, (-PI, 0), (-PI/2, PI/2)]
											+ [(0,0), (self.stand_pos[7]-PI/3, self.stand_pos[7]+PI/3), (self.stand_pos[8]-PI/3, self.stand_pos[8]+PI/3)]*2)
		pos_impl[2] = self._theta2_prime_hat(pos_impl[1])
		pos_impl[5] = self._theta2_prime_hat(pos_impl[4])
		# print(pos_impl[0])
		# print(f"theta1_prime1: {pos_impl[1]}; theta1_prime2: {pos_impl[4]}")
		# reach_limit = False

		if reach_limit:
			limit_cost = 0.1
		else:
			limit_cost = 0

		if self.leg_action_mode == "parallel_offset":
			self.leg_offsets[0] += action[4] * self.leg_offset_multiplier
			self.leg_offsets[1] += action[4] * self.leg_offset_multiplier
			self.leg_offsets[2] += action[5] * self.leg_offset_multiplier
			self.leg_offsets[3] += action[5] * self.leg_offset_multiplier
		elif self.leg_action_mode == "hips_offset":
			self.leg_offsets[0] += action[4] * self.leg_offset_multiplier
			self.leg_offsets[1] = 0
			self.leg_offsets[2] += action[5] * self.leg_offset_multiplier
			self.leg_offsets[3] = 0
		elif self.leg_action_mode == "knees_offset":
			self.leg_offsets[0] = 0
			self.leg_offsets[1] += action[4] * self.leg_offset_multiplier
			self.leg_offsets[2] = 0
			self.leg_offsets[3] += action[5] * self.leg_offset_multiplier
		elif self.leg_action_mode == "hips_knees_offset":
			self.leg_offsets[0] += action[4] * self.leg_offset_multiplier
			self.leg_offsets[1] += action[5] * self.leg_offset_multiplier
			self.leg_offsets[2] += action[6] * self.leg_offset_multiplier
			self.leg_offsets[3] += action[7] * self.leg_offset_multiplier
		elif self.leg_action_mode == "none":
			assert np.all(np.array(self.leg_offsets) == 0)

		self.leg_offsets = box(self.leg_offsets, [self.leg_offset_range]*4)[0]
		# print("[DEBUG] ACTION: ", action)
		# print("[DEBUG] LEG OFFSET: ", self.leg_offsets)

		for i in range(max_iter+1):
			timer7 = time.time()
			self._sub_step(i, max_iter, np.array(joints_pos_old), np.array(pos_impl),param=param, param_opt=self.param_opt, agent_action=action)  
			# print("STEPPING TAKES {:.6f} SEC".format(time.time() - timer7))
			self.sub_t += 1 
			timer7 = time.time()
			self._p.stepSimulation() 
			# print("SIMULATION TAKES {:.6f} SEC".format(time.time() - timer7))
			if self.real_time:
				time.sleep(0.001)
			if self.render:
				pass
				#self._move_camera()
			_ = self.get_full_state()  # for updating self.pitch and self.omegaBody

			if self.fast_error_update:
				self.p_error_buffer.append(self.pitch - self.pitch_ref)
				# print(f"[DEBUG] P ERROR BUFFER GOT: {self.pitch} - {self.pitch_ref} = {self.pitch - self.pitch_ref}")
				self.d_error_buffer.append(self.omegaBody[1])
				self.p_error_yaw_buffer.append(self.yaw)
				self.d_error_yaw_buffer.append(self.omegaBody[2])

				self.p_error_buffer = self.p_error_buffer[-self.error_buffer_size:]
				self.d_error_buffer = self.d_error_buffer[-self.error_buffer_size:]
				self.p_error_yaw_buffer = self.p_error_yaw_buffer[-self.error_buffer_size:]
				self.d_error_yaw_buffer = self.d_error_yaw_buffer[-self.error_buffer_size:]

				error_idx = int(self.latency / self.control_step)
				self.p_error = self.p_error_buffer[-(error_idx+1)] if error_idx+1 < len(self.p_error_buffer) else self.p_error_buffer[0]
				self.d_error = self.d_error_buffer[-(error_idx+1)] if error_idx+1 < len(self.d_error_buffer) else self.d_error_buffer[0]
				self.p_error_yaw = self.p_error_yaw_buffer[-(error_idx+1)] if error_idx+1 < len(self.p_error_yaw_buffer) else self.p_error_yaw_buffer[0]
				self.d_error_yaw = self.d_error_yaw_buffer[-(error_idx+1)] if error_idx+1 < len(self.d_error_yaw_buffer) else self.d_error_yaw_buffer[0]

				if self.latency == 0:
					assert(self.p_error == self.pitch - self.pitch_ref)
					assert(self.d_error == self.omegaBody[1])
					assert(self.p_error_yaw == self.yaw)
					assert(self.d_error_yaw == self.omegaBody[2])

				if self.progressing:
					self.error_buffer_size = 1000
					assert self.error_buffer_size >= 1000
					if self.sub_t > 200 and np.mean(np.absolute(self.d_error_buffer[-1000:])) < 0.2 and np.mean(np.absolute(self.p_error_buffer[-1000:])) < 0.1:

						if self.progressing_A_multiplier < self.param_opt[3]: # TODO: THIS IS INCONSITENT WITH C++ VERSION
							self.progressing_A_multiplier += 0.001
						# if self.progressing_delta_x < 0.1:
						# 	self.progressing_delta_x += 0.0001
						# print("UPGRADED!!!!!! A FACTOR: ", self.progressing_A_multiplier, "  DELAT_X: ", self.progressing_delta_x)
					else:
						pass
						# print("AVERAGE ERROR: ", np.mean(np.absolute(self.d_error_buffer[-100:])), "  A FACTOR: ", self.progressing_A_multiplier, "  DELAT_X: ", self.progressing_delta_x)


		self.leg_offsets_old = [i for i in self.leg_offsets]
		# self.leg_offsets_old = self.leg_offsets

		if not self.fast_error_update:  
			self.p_error_buffer.append(self.pitch - self.pitch_ref)
			self.d_error_buffer.append(self.omegaBody[1])
			self.p_error_yaw_buffer.append(self.yaw)
			self.d_error_yaw_buffer.append(self.omegaBody[2])

			self.p_error_buffer = self.p_error_buffer[-self.error_buffer_size:]
			self.d_error_buffer = self.d_error_buffer[-self.error_buffer_size:]
			self.p_error_yaw_buffer = self.p_error_yaw_buffer[-self.error_buffer_size:]
			self.d_error_yaw_buffer = self.d_error_yaw_buffer[-self.error_buffer_size:]

			error_idx = int(self.latency / (self.control_step*10)) 
			self.p_error = self.p_error_buffer[-(error_idx+1)] if error_idx+1 < len(self.p_error_buffer) else self.p_error_buffer[0]
			self.d_error = self.d_error_buffer[-(error_idx+1)] if error_idx+1 < len(self.d_error_buffer) else self.d_error_buffer[0]
			self.p_error_yaw = self.p_error_yaw_buffer[-(error_idx+1)] if error_idx+1 < len(self.p_error_yaw_buffer) else self.p_error_yaw_buffer[0]
			self.d_error_yaw = self.d_error_yaw_buffer[-(error_idx+1)] if error_idx+1 < len(self.d_error_yaw_buffer) else self.d_error_yaw_buffer[0]

		state, x, height = self._get_state()

		original_state = [i for i in state]  # FOR DEBUGGING

		# latency
		self.observations_buffer.append(state)
		self.observations_buffer = self.observations_buffer[-self.obs_buffer_size:]


		obs_idx = int(self.latency / (self.control_step*10)) # for example, if control_step is 0.002 and latency is 0.06, the agent should observe the observation 3 steps before 

		state = self.observations_buffer[-(obs_idx+1)] if obs_idx+1 < len(self.observations_buffer) else self.observations_buffer[0]
		state = np.array(state)


		if self.latency == 0:
			assert np.all(np.array(original_state) == state)
		
		action_cost = 0.25*abs(max(list(action[:4]), key=abs))

		reward = height - limit_cost + x - action_cost + 0.5
		self.t += 1

		if self.t > self.max_steps or height < 0.2 or abs(self.p_error) > 0.65 :  #self.pitch > -0.7  (error around 0.64066)  : #-0.27:
			# print("[DEBUG] ", self.t > 1000, ", ", height < 0.2, ", ", self.pitch > -0.7, ", PITCH: ", self.pitch)
			# print("[DEBUG] T: ",self.t , " HEIGHT: ", height, "P ERROR: ", self.p_error)
			# assert self.t > 1
			self.done = True
		# print(state)
		# print(self.t)
		assert np.array(state).size == self.observation_space.shape[0]
		info = {"pitch": self.pitch,
				"omega": self.omegaBody,
				"p_error": self.p_error,
				"d_error": self.d_error,
				"height": height
			   }

		# print("RPY: ", state[0:3])

		# print("[DEBUG I] MASS: ", self._p.getDynamicsInfo(self.dogId, -1)[0])
		# print("[DEBUG I] contact damping: ", self._p.getDynamicsInfo(self.planeId, -1)[8])
		# print("[DEBUG I] self.max_vel: ", self.max_vel)


		return state, reward, (self.done and not self.immortal), info

	
	def _get_state(self, pure_state=False):
		
		if self.version == 2 or self.state_mode == "h_body_arm":
			joints_state = self._p.getJointStates(self.dogId, self.motor_ids)
			joints_pos = [joints_state[i][0] for i in range(12)]
			torso_state = self._p.getBasePositionAndOrientation(self.dogId)
			torso_pos = [torso_state[0][i] for i in range(3)]
			height = torso_pos[2]
			torso_ori = [torso_state[1][i] for i in range(4)]

			# torso_ori_euler = self._p.getEulerFromQuaternion(torso_ori)
			torso_ori_euler = quat_to_YZX(torso_ori)
			# self.pitch = torso_ori_euler[1]  # some experiment update the pitch value here, which is not right

			get_velocity = self._p.getBaseVelocity(self.dogId)
			get_invert = self._p.invertTransform(torso_state[0], torso_state[1])
			get_matrix = self._p.getMatrixFromQuaternion(get_invert[1])
			torso_vel = [get_matrix[0] * get_velocity[1][0] + get_matrix[1] * get_velocity[1][1] + get_matrix[2] * get_velocity[1][2],
						 get_matrix[3] * get_velocity[1][0] + get_matrix[4] * get_velocity[1][1] + get_matrix[5] * get_velocity[1][2],
						 get_matrix[6] * get_velocity[1][0] + get_matrix[7] * get_velocity[1][1] + get_matrix[8] * get_velocity[1][2]]

			torso_ori_euler[0] += self.r_offset
			torso_ori_euler[1] += self.p_offset
			torso_ori_euler[2] += self.y_offset
			torso_vel[0] += self.w_r_offset
			torso_vel[1] += self.w_p_offset
			torso_vel[2] += self.w_y_offset

			# joint_pos_lite = [joints_pos[1], joints_pos[2], joints_pos[7],  joints_pos[8]]
			# joint_pos_lite = [joints_pos[1], joints_pos[7],  joints_pos[8], joints_pos[10], joints_pos[11]]
			joint_pos_lite = [joints_pos[0], joints_pos[1],  joints_pos[3], joints_pos[4]]
			
			if not pure_state:
				state = np.array([height] + torso_ori + torso_vel + joint_pos_lite + list(self.step_param)[:2])
			else:
				state = np.array([height] + torso_ori + torso_vel + joint_pos_lite)
		
			x = torso_pos[0]

		else:
			full_state = self.get_full_state()
			joints_pos = full_state[-12:]
			if self.state_mode == "body_arm_p":
				joint_pos_lite = [joints_pos[0], joints_pos[1],  joints_pos[3], joints_pos[4]]
				if not pure_state:
					state = np.array(list(full_state[6:12]) + joint_pos_lite + list(self.step_param)[:2])
				else:
					state = np.array(list(full_state[6:12]) + joint_pos_lite )
				x = full_state[0]
				height = full_state[2]

			elif self.state_mode == "body_arm":
				joint_pos_lite = [joints_pos[0], joints_pos[1],  joints_pos[3], joints_pos[4]]
				state = np.array(list(full_state[6:12]) + joint_pos_lite )
				x = full_state[0]
				height = full_state[2]

				if self.randomise and not self.only_randomise_dyn:
					half_range_imu = self.randomise / 100 * 2 #  1 --> 0.02
					half_range_encoder = self.randomise / 100 #  1 --> 0.01
					state = state + np.array(list(np.random.uniform(-half_range_imu, half_range_imu, size=6)) + list(np.random.uniform(-half_range_encoder, half_range_encoder, size=4)) )


			elif self.state_mode == "body_arm_leg_full_p":
				joint_pos_lite = [joints_pos[0], joints_pos[1],  joints_pos[3], joints_pos[4], joints_pos[7], joints_pos[8], joints_pos[10], joints_pos[11]]
				if not pure_state:
					state = np.array(list(full_state[6:12]) + joint_pos_lite + list(self.step_param)[:2])
				else:
					state = np.array(list(full_state[6:12]) + joint_pos_lite )
				x = full_state[0]
				height = full_state[2]

			elif self.state_mode == "body_arm_leg_full":
				joint_pos_lite = [joints_pos[0], joints_pos[1],  joints_pos[3], joints_pos[4], joints_pos[7], joints_pos[8], joints_pos[10], joints_pos[11]]
				state = np.array(list(full_state[6:12]) + joint_pos_lite )
				assert state.size == 3+3+4+4
				x = full_state[0]
				height = full_state[2]

				if self.randomise and not self.only_randomise_dyn:
					half_range_imu = self.randomise / 100 * 2 #  1 --> 0.02
					half_range_encoder = self.randomise / 100 #  1 --> 0.01
					state = state + np.array(list(np.random.uniform(-half_range_imu, half_range_imu, size=6)) + list(np.random.uniform(-half_range_encoder, half_range_encoder, size=8)))

			elif self.state_mode == "body_arm_leg_full_i":
				joint_pos_lite = [joints_pos[0], joints_pos[1],  joints_pos[3], joints_pos[4], joints_pos[7], joints_pos[8], joints_pos[10], joints_pos[11]]
				state = np.array(list(full_state[6:12]) + joint_pos_lite )
				assert state.size == 3+3+4+4
				x = full_state[0]
				height = full_state[2]

				if self.randomise and not self.only_randomise_dyn:
					half_range_imu = self.randomise / 100 * 2 #  1 --> 0.02
					half_range_encoder = self.randomise / 100 #  1 --> 0.01
					state = state + np.array(list(np.random.uniform(-half_range_imu, half_range_imu, size=6)) + list(np.random.uniform(-half_range_encoder, half_range_encoder, size=8)))

				state = np.array(list(state) + [self.leg_state_indicator]) # DEVEVLOPING!!

			elif self.state_mode == "body_arm_leg_full_il":
				joint_pos_lite = [joints_pos[0], joints_pos[1],  joints_pos[3], joints_pos[4], joints_pos[7], joints_pos[8], joints_pos[10], joints_pos[11]]
				state = np.array(list(full_state[6:12]) + joint_pos_lite )
				assert state.size == 3+3+4+4
				x = full_state[0]
				height = full_state[2]

				if self.randomise and not self.only_randomise_dyn:
					half_range_imu = self.randomise / 100 * 2 #  1 --> 0.02
					half_range_encoder = self.randomise / 100 #  1 --> 0.01
					state = state + np.array(list(np.random.uniform(-half_range_imu, half_range_imu, size=6)) + list(np.random.uniform(-half_range_encoder, half_range_encoder, size=8)))

				state = np.array(list(state) + self.leg_state_indicators) # DEVEVLOPING!!

			
			self.states.append(state)
			# print("[DEBUG] ORIGINAL STATE: ", state)

			if len(self.states) > self.num_history_observation + 1:
				self.states = self.states[1:]
			elif len(self.states) < self.num_history_observation + 1:
				while len(self.states) < self.num_history_observation + 1:
					self.states.append(self.states[-1])

			assert len(self.states) == self.num_history_observation + 1

			if self.num_history_observation == 0:
				assert np.all(np.array(self.states).flatten() == state)

			state = np.array(self.states).flatten()



		return state, x, height

	def _update_step_param_buffered(self, only_legs=True, action=None):
		# only update the first two parametersied actions parameters
		if self.action_mode == "partial":
			# if param is not None:
			# 	# parameterised actions are already calculated by the agent 
			# 	if not self.arm_pd_control:
			# 		assert len(param) == 2
			# 	else:
			# 		assert len(param) == 3 # the additional parameter is to control bending forward or backward
			# 	self.step_param_buffered = [i for i in param]
			# elif param_opt is not None or self.tuner_enable:

			assert param_opt is not None
			# only the control parameters are given 

			self.step_param_buffered[1] = min(max(self.param_opt[0] + self.param_opt[1] * self.p_error + self.param_opt[2] * self.d_error, self.B_range[0]), self.B_range[1])
			self.step_param_buffered[0] = min(max(self.step_param_buffered[1] * self.param_opt[3], self.A_range[0]), self.A_range[1])
			
			# else:
			# 	raise ValueError('Optimised parameters are not provided in partial mode.')

		elif self.action_mode == "whole":
			self.step_param_buffered[0] = (action[4]+1)/2*0.9+0.1  # -1~1 -> 0.1~1
			self.step_param_buffered[1] = (action[5]+1)/2*0.09+0.01  # -1~1 -> 0.01~0.1

		elif self.action_mode == "residual":
			# assert param is not None  # param_opt mode is not implemented yet
			if self.leg_action_mode == "parameter":  # additional actions from the agent is parameters for parameterised control
				param = [0, 0]
				param[0] = min(max(self.param_opt[0] + self.param_opt[1] * self.p_error + self.param_opt[2] * self.d_error, self.B_range[0]), self.B_range[1])
				param[1] = min(max(self.step_param_buffered[1] * self.param_opt[3], self.A_range[0]), self.A_range[1])
				self.step_param_buffered[0] = param[0] + action[4]*self.residual_multiplier  # -1~1 -> -0.5~0.5
				self.step_param_buffered[1] = param[1] + action[5]*self.residual_multiplier  # -1~1 -> -0.5~0.5
				self.step_param_buffered[0] = max(min(self.step_param_buffered[0], self.A_range[1]), self.A_range[0])
				self.step_param_buffered[1] = max(min(self.step_param_buffered[1], self.B_range[1]), self.B_range[0])
			else:
				self.step_param_buffered[1] = min(max(self.param_opt[0] + self.param_opt[1] * self.p_error + self.param_opt[2] * self.d_error, self.B_range[0]), self.B_range[1])
				# print(f"self.step_param_buffered[1] = min(max({self.param_opt[0]} + {self.param_opt[1]} * {self.p_error} + {self.param_opt[2]} * {self.d_error}, {self.B_range[0]}), {self.B_range[1]}) = {self.step_param_buffered[1]}")
				# self.step_param_buffered[0] = min(max(self.step_param_buffered[1] * self.param_opt[3], self.A_range[0]), self.A_range[1])
				if not self.progressing:
					self.step_param_buffered[0] = min(max(self.step_param_buffered[1] * self.param_opt[3], self.A_range[0]), self.A_range[1])
					# print(f"self.step_param_buffered[0] = min(max({self.step_param_buffered[1]} * {self.param_opt[3]}, {self.A_range[0]}), {self.A_range[1]}) = {self.step_param_buffered[0]}")
				else:
					assert self.A_range[1] == float("inf")
					# print(self.step_param_buffered[1] * self.progressing_A_multiplier)
					# print(f"max(self.step_param_buffered[1] * self.progressing_A_multiplier, self.A_range[0]) = {max(self.step_param_buffered[1] * self.progressing_A_multiplier, self.A_range[0])}")
					# print(f"min({max(self.step_param_buffered[1] * self.progressing_A_multiplier, self.A_range[0])}, self.A_range[1]) = {self.step_param_buffered[0]}")
					self.step_param_buffered[0] = min(max(self.step_param_buffered[1] * self.progressing_A_multiplier, self.A_range[0]), self.A_range[1])

		self.step_param_buffer.append(self.step_param_buffered)

		if not hasattr(self, "step_param"):
			self.step_param = [i for i in self.step_param_buffered]

	def _sub_step(self, curr_iter, max_iter, ini_pos, fin_pos, param=None, param_opt=None, leg_action=None, agent_action=None):

		if curr_iter <= max_iter:
			b = curr_iter/max_iter;
			a = 1 - b;
			inter_pos = a * ini_pos + b * fin_pos;

			leg_offsets = a * np.array(self.leg_offsets_old) + b * np.array(self.leg_offsets)
			# print(f"leg_offsets = {a} * {np.array(self.leg_offsets_old)} + {b} * {np.array(self.leg_offsets)} = {leg_offsets}")
		else:
			inter_pos = fin_pos
			leg_offsets = self.leg_offsets

		if not self.leg_bootstrapping:
			assert(np.all(np.array(leg_offsets) <= self.leg_offset_range[1]) and np.all(np.array(leg_offsets) >= self.leg_offset_range[0]))

		if self.external_force:
			max_rand_force = self.external_force*10
			self._p.applyExternalForce(self.dogId, -1, [np.random.uniform(-max_rand_force, max_rand_force), 0, np.random.uniform(-max_rand_force, max_rand_force)], [0, 0, 0], flags=self._p.LINK_FRAME)


		if param is not None or param_opt is not None:

			self._update_step_param_buffered(action=agent_action)  # param_opt -> step_param
			a_sin = self.step_param[0]
			b_sin = self.step_param[1]

			theta2r_delta = - a_sin*(math.sin(b_sin*self.sub_t-PI/2)/2+0.5)
			theta2l_delta = - a_sin*(-math.sin(b_sin*self.sub_t-PI/2)/2+0.5)

			# print("[DEBUG] self.param_opt: ", self.param_opt)
			# print("[DEBUG] self.a_sin: ", a_sin)
			# print("[DEBUG] self.b_sin: ", b_sin)

			if not (self.gait == "line" or self.gait == "none"):  
				# new gait, the robot will step forward
				self.delta_x = self.param_opt[8] # SOME EXPERIEMNT FORGOT THIS UPDATING !!!


				period_half = PI/max(b_sin, 0.00001)   # in this case, period_half=100  ->  b_sin=0.0314
				sub_sub_t = self.sub_t%(period_half+period_half)
				# print("[DEBUG] half period: ", period_half, "  sub_t: ", sub_sub_t, " B: ", b_sin)

				# if sub_sub_t >= period_half and sub_sub_t-1 < period_half:

					# # TESTING FEEDBACK CONTROL
					# self.delta_x = self.param_opt[8] + self.p_error*0.04 + self.d_error*0.04
					# print(f"self.delta_x = {self.param_opt[8]} + {self.p_error*0.01} + {self.d_error*0.01} = {self.delta_x}")



				if self.gait == "sine":
					if sub_sub_t < period_half:
						x_togo_r = self.x_original + (self.delta_x/period_half)*sub_sub_t
						y_togo_r = self.y_original + a_sin * math.sin(PI/period_half*sub_sub_t)

						if not sub_sub_t == self.sub_t:
							x_togo_l = self.x_original + self.delta_x - sub_sub_t * self.delta_x/period_half
							y_togo_l = self.y_original
						else:
							x_togo_l = self.x_original
							y_togo_l = self.y_original
						
					else:
						x_togo_r = self.x_original + self.delta_x - (sub_sub_t-period_half) * self.delta_x/period_half
						y_togo_r = self.y_original
						x_togo_l = self.x_original + (self.delta_x/period_half)*(sub_sub_t-period_half)
						y_togo_l = self.y_original + a_sin * math.sin(PI/period_half*(sub_sub_t-period_half))

					self.leg_state_indicators = [(self.delta_x/period_half)*sub_sub_t, y_togo_l]  # DEVELOPING

				elif self.gait == "rose":

					a_rose = max(self.delta_x, 0.00001)
					k_rose = 4*a_sin/a_rose

					if sub_sub_t < period_half:
						th = (PI/4)/period_half*(period_half-sub_sub_t)
						x_togo_r = self.x_original + a_rose * math.cos(2*th) * math.cos(th)
						y_togo_r = self.y_original + k_rose * a_rose * math.cos(2*th) * math.sin(th)

						if not sub_sub_t == self.sub_t:
							x_togo_l = self.x_original + self.delta_x - sub_sub_t * self.delta_x/period_half
							y_togo_l = self.y_original
						else:
							x_togo_l = self.x_original
							y_togo_l = self.y_original
						
					else:
						th = (PI/4)/period_half*(period_half-(sub_sub_t-period_half))
						x_togo_r = self.x_original + self.delta_x - (sub_sub_t-period_half) * self.delta_x/period_half
						y_togo_r = self.y_original
						x_togo_l = self.x_original + a_rose * math.cos(2*th) * math.cos(th)
						y_togo_l = self.y_original + k_rose * a_rose * math.cos(2*th) * math.sin(th)

					self.leg_state_indicators = [th, y_togo_l]  # DEVELOPING

				elif self.gait == "triangle":

					x_0, y_0 = self._FK(self.theta1, self.theta2)
					x_1, y_1 = self._FK(self.theta1-(-a_sin), self.theta2+(-a_sin))
					x_2, y_2 = x_0+self.delta_x, y_0

					if sub_sub_t < period_half:
						if b_sin*sub_sub_t < PI/2:  # == sub_sub_t < period_half/2:
							# print(f"{b_sin*sub_sub_t} --> {PI/2}")
							x_togo_r = x_0 + (math.sin(2*b_sin*sub_sub_t-PI/2)+1)/2*(x_1-x_0)
							y_togo_r = y_0 + (math.sin(2*b_sin*sub_sub_t-PI/2)+1)/2*(y_1-y_0)
						else:
							# print(f"{b_sin*sub_sub_t} --> {PI}")
							x_togo_r = x_1 + (-math.sin(2*b_sin*sub_sub_t-PI/2)/2+0.5)*(x_2-x_1)
							y_togo_r = y_1 + (-math.sin(2*b_sin*sub_sub_t-PI/2)/2+0.5)*(y_2-y_1)

						if not sub_sub_t == self.sub_t:
							x_togo_l = self.x_original + self.delta_x - sub_sub_t * self.delta_x/period_half
							y_togo_l = self.y_original
						else:
							x_togo_l = self.x_original
							y_togo_l = self.y_original
						
					else:
						x_togo_r = self.x_original + self.delta_x - (sub_sub_t-period_half) * self.delta_x/period_half
						y_togo_r = self.y_original
						assert self.y_original == y_2 and y_2 == y_0

						if b_sin*(sub_sub_t-period_half) < PI/2:
							x_togo_l = x_0 + (math.sin(2*b_sin*(sub_sub_t-period_half)-PI/2)/2+0.5)*(x_1-x_0)
							y_togo_l = y_0 + (math.sin(2*b_sin*(sub_sub_t-period_half)-PI/2)/2+0.5)*(y_1-y_0)
						else:
							x_togo_l = x_1 + (-math.sin(2*b_sin*(sub_sub_t-period_half)-PI/2)/2+0.5)*(x_2-x_1)
							y_togo_l = y_1 + (-math.sin(2*b_sin*(sub_sub_t-period_half)-PI/2)/2+0.5)*(y_2-y_1)
			

					self.leg_state_indicators = [b_sin*(sub_sub_t-period_half), y_togo_l]  # DEVELOPING

				self.leg_state_indicator = y_togo_r  # develeoping

				inter_pos[7], inter_pos[8] = self._IK(x_togo_r, y_togo_r)
				inter_pos[10], inter_pos[11] = self._IK(x_togo_l, y_togo_l)

				if self.sub_step_callback is not None:
					self.sub_step_callback(t=self.sub_t, leg_basic_action=[inter_pos[7],inter_pos[8],inter_pos[10],inter_pos[11]], leg_offset=leg_offsets,
										   leg_final_action=[inter_pos[7] - leg_offsets[0], inter_pos[8] + leg_offsets[1], inter_pos[10] - leg_offsets[2], inter_pos[11] + leg_offsets[3]],
										   arm_final_action=[inter_pos[0], inter_pos[1], inter_pos[3], inter_pos[4]])  
				# TODO: MAKE A MORE GENERAL AND INFORMATIVE VERSION

				inter_pos[7] -= leg_offsets[0]
				inter_pos[8] += leg_offsets[1]
				inter_pos[10] -= leg_offsets[2]
				inter_pos[11] += leg_offsets[3]


				# # TESTING CONTROL LEG ABD
				# inter_pos[6] = -abs(self.step_param_buffered[2])*2
				# inter_pos[9] = abs(self.step_param_buffered[2])*2


			# elif self.version == 2 or self.state_mode == "h_body_arm":

			# 	if (b_sin*self.sub_t < PI):
			# 		inter_pos[8] = self.theta2 + theta2r_delta
			# 		inter_pos[7] = self._theta1_hat(inter_pos[8])
			# 		inter_pos[10] = self.theta1
			# 		inter_pos[11] = self.theta2
			# 	else:
			# 		inter_pos[8] = self.theta2 + theta2r_delta
			# 		inter_pos[7] = self._theta1_hat(inter_pos[8])
			# 		inter_pos[11] = self.theta2 + theta2l_delta
			# 		inter_pos[10] = self._theta1_hat(inter_pos[11])

			elif self.gait == "line":
				assert self.version == 3  # version 2 is not supported anymore
				period_half = PI/max(b_sin, 0.00001)

				if (b_sin*self.sub_t < PI):
					inter_pos[8] =  self.theta2 + theta2r_delta
					inter_pos[7] = self.theta1 - theta2r_delta
					inter_pos[10] = self.theta1
					inter_pos[11] = self.theta2
				else:
					inter_pos[8] = self.theta2 + theta2r_delta + leg_offsets[1]
					inter_pos[7] = self.theta1 - theta2r_delta - leg_offsets[0]
					inter_pos[11] = self.theta2 + theta2l_delta + leg_offsets[3]
					inter_pos[10] = self.theta1 - theta2l_delta - leg_offsets[2]

				self.leg_state_indicator = theta2r_delta
				self.leg_state_indicators = [self.sub_t%(period_half+period_half), theta2r_delta]

			elif self.gait == "none":
				inter_pos[8] = self.theta2  + leg_offsets[1]
				inter_pos[7] = self.theta1 + leg_offsets[0]
				inter_pos[11] = self.theta2 + leg_offsets[3]
				inter_pos[10] = self.theta1 + leg_offsets[2]


			sin_value = math.sin(b_sin*self.sub_t-PI/2)
			# print(f"sin({b_sin}t) = {sin_value}   buffered frequency: {self.step_param_buffered[1]}")
			# print(f"frequency: {b_sin}  buffered frequency: {self.step_param_buffered[1]}")
			assert b_sin == self.step_param[1]
			if self.sin_value*sin_value <= 0:
				# print("==========  direction changed!  ==========")
				buffer_avg = np.mean(self.step_param_buffer, axis=0)
				self.step_param[0] = buffer_avg[0]
				self.step_param[1] = buffer_avg[1]
				self.step_param_buffer = []

				# the position of this part of code may be wrong

				# print(f"P ERROR: {self.p_error}  D ERROR: {self.d_error}")
				# if not hasattr(self, "old_sub_t"):
				# 	self.old_sub_t = self.sub_t
				# print(f"HALF A DURATION IS {time.time() - self.timer0} S, {self.sub_t - self.old_sub_t} TIME STEPS")
				# self.old_sub_t = self.sub_t
				# self.timer0 = time.time()

			# print(f"CONTROL FREQUENCY IS {time.time() - self.timer0} S")
			# self.timer0 = time.time()
			# print(f"SIN ({b_sin} x {self.sub_t} - PI/2) = ({(b_sin*self.sub_t-PI/2)/PI}PI)")


			self.sin_value = sin_value

			# inter_pos[7] = self.stand_pos[7] + a_sin*math.sin(b_sin*self.sub_t) 
			# inter_pos[8] = self.stand_pos[8] - a_sin*math.sin(b_sin*self.sub_t) 
			# # inter_pos[7] = self._theta1_hat(inter_pos[8])
			# inter_pos[10] = self.stand_pos[10] - a_sin*math.sin(b_sin*self.sub_t) 
			# inter_pos[11] = self.stand_pos[11] + a_sin*math.sin(b_sin*self.sub_t) 
			# # inter_pos[10] = self._theta1_hat(inter_pos[11])

		

		for i, j in enumerate(self.motor_ids_l):
			pos = inter_pos[self.motor_ids.index(j)]
			self._p.setJointMotorControl2(self.dogId, j, self._p.POSITION_CONTROL, pos, force=self.motor_force_l, maxVelocity=self.max_vel)
		for i, j in enumerate(self.motor_ids_r):
			pos = inter_pos[self.motor_ids.index(j)]
			self._p.setJointMotorControl2(self.dogId, j, self._p.POSITION_CONTROL, pos, force=self.motor_force_r, maxVelocity=self.max_vel)

	def _theta2_prime_hat(self, theta1_prime):
		# if theta1_prime > -PI/2 - 0.63962:
		# 	theta2_prime = -0.58*theta1_prime
		# else:
		# 	theta2_prime = 1.3738*theta1_prime+4.3160
		if theta1_prime > -2.3022:
			theta2_prime = -0.6354*theta1_prime
		else:
			theta2_prime = 1.7430*theta1_prime+5.4759
		# print(f"{theta1_prime} --> {theta2_prime}")
		
		return theta2_prime

	def _theta1_hat(self, theta2):

		# k*a*math.cos(alpha) + c*math.cos(gamma) - b*math.cos(beta) = l/2

		# alpha + beta + theta1 = PI/2
		# beta + gamma + theta2 = PI
		# beta = PI - theta2 - gamma
		# alpha = -PI/2 - theta1 + theta2 + gamma

		gamma, k, a, b, c, l = self.gamma, self.k, self.a, self.b, self.c, self.l

		neg_theta2 = -theta2
		try:
			ans = -PI/2 + neg_theta2 + gamma - math.acos((l/2 + b*math.cos(PI - neg_theta2 - gamma) - c*math.cos(gamma)) / k / a )
			# print("THE RESULT OF ACOS IS ", math.acos((l/2 + b*math.cos(PI - theta2 - gamma) - c*math.cos(gamma)) / k / a ))
		except ValueError:
			print("FAIL TO CALCUATE THE ACOS OF ", (l/2 + b*math.cos(PI - neg_theta2 - gamma) - c*math.cos(gamma)) / k / a )
			ans = PI/2 #np.nan
		return ans

	def _FK(self, th1, th2):
		gamma, k, a, b, c, l = self.gamma, self.k, self.a, self.b, self.c, self.l
		return c*math.cos(th1) + b*math.cos(th1+th2), c*math.sin(th1) + b*math.sin(th1+th2)


	def _IK(self, x, y):
		gamma, k, a, b, c, l = self.gamma, self.k, self.a, self.b, self.c, self.l

		def th2_ik(x, y):
			try:
				th2 = math.atan2( - math.sqrt(1 - ((x**2 + y**2 - c**2 - b**2) / (2*c*b))**2), (x**2 + y**2 - c**2 - b**2) / (2*c*b))
				# th2 = math.atan2( + math.sqrt(1 - ((x**2 + y**2 - c**2 - b**2) / (2*c*b))**2), (x**2 + y**2 - c**2 - b**2) / (2*c*b))
			except ValueError:
				th2 = 0
			return th2

		def th1_ik(th2, x, y):
			k_1 = c + b*math.cos(th2)
			k_2 = b*math.sin(th2)
			th1 = math.atan2(y, x) - math.atan2(k_2, k_1)
			return th1

		theta2_ik = th2_ik(x, y)
		theta1_ik = th1_ik(float(theta2_ik), x, y)

		return theta1_ik, theta2_ik


	def _move_camera(self):
		dogPos, _ = self._p.getBasePositionAndOrientation(self.dogId)
		self._p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=dogPos)


	def _set_dynamics(self, mass_body=6.841, mass_abduct=0.550, thigh_shank_mass=0.200, toe_mass=0.0273, lateralFriction_shank=0.351, lateralFriction_toe=0.512, 
					 maxJointVelocity=20.9440, contactStiffness=2000, contactDamping=20, linearDamping=0.04, angularDamping=0.04, jointDamping=0.01, max_force=18, rl_force_offset=0,
					 w_r_offset=0, w_p_offset=0, w_y_offset=0, v_x_offset=0, v_y_offset=0, v_z_offset=0, latency=0, control_step=0.002):

		# print("[DEBUG] contactStiffness: ", contactStiffness)
		abduct_links = [0, 4, 8, 12]
		thigh_links = [1, 5, 9, 13]
		shank_links = [2, 6, 10, 14]
		toe_links = [3, 7, 11, 15]

		self.custom_body_mass = mass_body
		if self.fix_body:
			mass_body = 0

		# p.changeDynamics(self.dogId, 0, mass=666, linearDamping=0, angularDamping=0, jointDamping=0)
		# p.changeDynamics(self.dogId, 0, mass=0.666, maxJointVelocity=20, linearDamping=0, angularDamping=0, jointDamping=0)
		self._p.changeDynamics(self.dogId, -1, mass=mass_body, linearDamping=linearDamping, angularDamping=angularDamping, jointDamping=jointDamping)
		# return 0

		for link in abduct_links:
			self._p.changeDynamics(self.dogId, link, mass=mass_abduct, 
				# linearDamping=linearDamping, angularDamping=angularDamping, 
				jointDamping=jointDamping)
		for link in thigh_links:
			self._p.changeDynamics(self.dogId, link, mass=thigh_shank_mass, 
				# linearDamping=linearDamping, angularDamping=angularDamping, 
				jointDamping=jointDamping)

		for link in shank_links:
			self._p.changeDynamics(self.dogId, link, mass=thigh_shank_mass, lateralFriction=lateralFriction_shank, 
				# contactStiffness=contactStiffness, contactDamping=contactDamping, 
				# linearDamping=linearDamping, angularDamping=angularDamping, 
				jointDamping=jointDamping)

		for link in toe_links:
			self._p.changeDynamics(self.dogId, link, mass=toe_mass, lateralFriction=lateralFriction_toe, 
				# ccontactStiffness=contactStiffness, contactDamping=contactDamping, l
				# linearDamping=linearDamping, angularDamping=angularDamping, 
				# jointDamping=jointDamping
				)

		self._p.changeDynamics(self.planeId, -1, contactStiffness=contactStiffness, contactDamping=contactDamping)

		self.motor_force_r = max_force
		self.motor_force_l = max_force + rl_force_offset
		self.max_vel = maxJointVelocity
		self.r_offset, self.p_offset, self.y_offset = 0, 0, 0
		self.x_offset, self.yy_offset, self.z_offset = 0, 0, 0
		self.w_r_offset, self.w_p_offset, self.w_y_offset = w_r_offset, w_p_offset, w_y_offset
		self.v_x_offset, self.v_y_offset, self.v_z_offset = v_x_offset, v_y_offset, v_z_offset

		self.latency = latency
		self.control_step = control_step

		self._p.setPhysicsEngineParameter(fixedTimeStep=self.control_step)


	def set_dynamics(self, custom_dynamics):
		self.dynamcis_setting = {**self.default_dynamcis, **custom_dynamics}


	def debug_dynamics(self):


		# print("[DEBUG] contactStiffness: ", contactStiffness)
		abduct_links = [0, 4, 8, 12]
		thigh_links = [1, 5, 9, 13]
		shank_links = [2, 6, 10, 14]
		toe_links = [3, 7, 11, 15]

		mass_body=6.841
		linearDamping=0.04
		angularDamping=0.04
		jointDamping=0.01
		lateralFriction_shank=0.351
		lateralFriction_toe=0.512
		contactStiffness=2000
		contactDamping=20

		# self._p.changeDynamics(self.dogId, 0, mass=666, linearDamping=0, angularDamping=0, jointDamping=0)
		# self._p.changeDynamics(self.dogId, 0, mass=0.666, maxJointVelocity=20, linearDamping=0, angularDamping=0, jointDamping=0)

		self._p.changeDynamics(self.dogId, -1, linearDamping=linearDamping, angularDamping=angularDamping, jointDamping=jointDamping)

		

		for link in abduct_links:
			self._p.changeDynamics(self.dogId, link, linearDamping=linearDamping, angularDamping=angularDamping, jointDamping=jointDamping)
		for link in thigh_links:
			self._p.changeDynamics(self.dogId, link, linearDamping=linearDamping, angularDamping=angularDamping, jointDamping=jointDamping)

		for link in shank_links:
			self._p.changeDynamics(self.dogId, link, lateralFriction=lateralFriction_shank, 
				# contactStiffness=contactStiffness, contactDamping=contactDamping, 
				linearDamping=linearDamping, angularDamping=angularDamping, jointDamping=jointDamping)

		for link in toe_links:
			self._p.changeDynamics(self.dogId, link, lateralFriction=lateralFriction_toe, 
				# contactStiffness=contactStiffness, contactDamping=contactDamping, 
				linearDamping=linearDamping, angularDamping=angularDamping, jointDamping=jointDamping)
		self._p.changeDynamics(self.planeId, -1, contactStiffness=contactStiffness, contactDamping=contactDamping)
		return 0
		

	def get_full_state(self):
		
		joints_state = self._p.getJointStates(self.dogId, self.motor_ids)
		joints_pos = [joints_state[i][0] for i in range(12)]
		pos, quat = self._p.getBasePositionAndOrientation(self.dogId)
		pos = [pos[i] for i in range(3)]
		height = pos[2]
		torso_ori = [quat[i] for i in range(4)]

		# torso_ori_euler = self._p.getEulerFromQuaternion(torso_ori)
		torso_yzx = quat_to_YZX(torso_ori)
		self.pitch = torso_yzx[1]
		self.yaw = torso_yzx[2]

		vWorld, omegaWorld = self._p.getBaseVelocity(self.dogId)
		p_invert, quat_invert = self._p.invertTransform(pos, quat)
		get_matrix = self._p.getMatrixFromQuaternion(quat_invert)
		omegaBody = [get_matrix[0] * omegaWorld[0] + get_matrix[1] * omegaWorld[1] + get_matrix[2] * omegaWorld[2], 
					 get_matrix[3] * omegaWorld[0] + get_matrix[4] * omegaWorld[1] + get_matrix[5] * omegaWorld[2],
					 get_matrix[6] * omegaWorld[0] + get_matrix[7] * omegaWorld[1] + get_matrix[8] * omegaWorld[2]]
		# Probably right
		self.omegaBody = omegaBody
		vWorld = [vWorld[i] for i in range(3)]
		omegaWorld = [omegaWorld[i] for i in range(3)]

		self.x_offset += self.v_x_offset*0.002
		self.yy_offset += self.v_y_offset*0.002
		self.z_offset += self.v_z_offset*0.002
		self.r_offset += self.w_r_offset*0.002
		self.p_offset += self.w_p_offset*0.002
		self.y_offset += self.w_y_offset*0.002


		pos[0] += self.x_offset
		pos[1] += self.yy_offset
		pos[2] += self.z_offset
		vWorld[0] += self.v_x_offset
		vWorld[1] += self.v_y_offset
		vWorld[2] += self.v_z_offset
		torso_yzx[0] += self.r_offset
		torso_yzx[1] += self.p_offset
		torso_yzx[2] += self.y_offset
		omegaWorld[0] += self.w_r_offset
		omegaWorld[1] += self.w_p_offset
		omegaWorld[2] += self.w_y_offset

		state = np.array(pos + vWorld + torso_yzx + omegaWorld + joints_pos)  # 3 + 3 + 3 + 3 + 12 = 24

		return state

	def reset_jpos(self, jpos):
		assert len(jpos) == 12
		for j, pos in zip(self.motor_ids, jpos):
			self._p.resetJointState(self.dogId, j, pos, 0)
		if self.debug_tuner_enable:
			self.debug_values = [self._p.readUserDebugParameter(tunner) for tunner in self.debug_tuners]

	def step_jpos(self, jpos=None):
		if jpos is not None:
			assert len(jpos) == 12
			for i, pos in zip(self.motor_ids, jpos):
				self._p.setJointMotorControl2(self.dogId, i, self._p.POSITION_CONTROL, pos, force=self.motor_force_r, maxVelocity=self.max_vel)
		self._p.stepSimulation() 

	def update_lcm_state(self):

		joints_state = self._p.getJointStates(self.dogId, self.motor_ids)
		pos, quat = self._p.getBasePositionAndOrientation(self.dogId)
		vWorld, omegaWorld = self._p.getBaseVelocity(self.dogId)
		p_invert, quat_invert = self._p.invertTransform(pos, quat)
		get_matrix = self._p.getMatrixFromQuaternion(quat_invert)
		omegaBody = [get_matrix[0] * omegaWorld[0] + get_matrix[1] * omegaWorld[1] + get_matrix[2] * omegaWorld[2], 
					 get_matrix[3] * omegaWorld[0] + get_matrix[4] * omegaWorld[1] + get_matrix[5] * omegaWorld[2],
					 get_matrix[6] * omegaWorld[0] + get_matrix[7] * omegaWorld[1] + get_matrix[8] * omegaWorld[2]]

		self.lcm_state = self.LcmState()
		self.lcm_state.p = [pos[i] for i in range(3)]
		self.lcm_state.vWorld = [i for i in vWorld]
		self.lcm_state.vBody = [0]*3
		self.lcm_state.vRemoter = [0]*3
		self.lcm_state.rpy = [0]*3
		self.lcm_state.omegaBody = [i for i in omegaBody]
		self.lcm_state.omegaWorld = [i for i in omegaWorld]
		self.lcm_state.quat = [quat[3], quat[0], quat[1], quat[2]]   # quat [x, y, z, w] -->  lcm.quat [w, x, y, z]
		self.lcm_state.aBody = [0]*3
		self.lcm_state.aWorld = [0]*3

		self.lcm_state.q = [joints_state[i][0] for i in range(12)]
		self.lcm_state.qd = [0]*12
		self.lcm_state.p_leg = [0]*12
		self.lcm_state.v = [0]*12
		self.lcm_state.tau_est = [0]*12

	class LcmState(object):
		def __init__(self):
			self.p = []
			self.vWorld = []
			self.vBody = []
			self.vRemoter = []
			self.rpy = []
			self.omegaBody = []
			self.omegaWorld = []
			self.quat = []
			self.aBody = []
			self.aWorld = []
			self.q = []
			self.qd = []
			self.p_leg = []
			self.v = []
			self.tau_est = []

	def get_joins_pos(self):
		joints_state = self._p.getJointStates(self.dogId, self.motor_ids)
		return [joints_state[i][0] for i in range(12)]



def human_optimiser():

	DYN_CONFIG = {'lateralFriction_toe': 0.7, #1, # 0.6447185797960826, 
				  'lateralFriction_shank': 0.5, #0.6447185797960826  *  (0.351/0.512),
				  'contactStiffness': 2592, #2729, # 4173, #2157.4863390669952, 1615?
				  'contactDamping': 450, #414, #160, # 122, #32.46233575737161,    150?
				  'linearDamping': 0.03111169082496665, 
				  'angularDamping': 0.04396695866661371, 
				  'jointDamping': 0.03118494025640309, 
				  "max_force": 12, #10,
				  "mass_body": 9.5 #10.5
				  }

	LEG_ACTION = ["none", "parallel_offset", "hips_offset", "knees_offset", "hips_knees_offset"][2]

	env_args = {"render": True, "fix_body": False, "real_time": True, "immortal": False, 
				"version": 3, "normalised_abduct": True, "mode": "stand", "debug_tuner_enable" : True, "action_mode":"residual", "state_mode":"body_arm", "leg_action_mode":LEG_ACTION,
				"tuner_enable": True, "action_tuner_enable": True, "A_range": (0.01, 0.4), "gait": "rose", "custom_dynamics": DYN_CONFIG, "param_opt": [0.0251, 0.0000, 0.0000, 1.6714, 0.0715, 0.0923, 0.0860, 0.0858, 0.0124],
				"fast_error_update": True, "arm_pd_control": True, "custom_robot": {"k": 0.72}, "progressing": False, "max_steps": 1000,
				"leg_offset_multiplier": 0.1, "action_multiplier": 0.01, "randomise": 3} # , "progressing_agent_factor": True

	dog = Dog(**env_args)
	# dog._set_dynamics(**DYN_CONFIG)

	n_epi = 0
	r_ = 0
	while True:
		done = False
		dog.reset()
		n_step = 0
		while not done:
			s, r, done, info =  dog.step([0]*6, param_opt=[0]*9)
			n_step += 1
			r_ += r
			#print("STEP ", n_step) 
			print("RETURN ", r_) 
		n_epi += 1


def simple_debug():

	def sub_step_callback(t, leg_basic_action, leg_offset, leg_final_action, arm_final_action):
		pass
		# print(f"SUB STEP {t}\n")
		# print("BASIC ACTION: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]\n".format(leg_basic_action[0], leg_basic_action[1], leg_basic_action[2], leg_basic_action[3]))
		# print("OFFSET: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]\n".format(leg_offset[0], leg_offset[1], leg_offset[2], leg_offset[3]))
		# print("FINAL ACTION: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]\n".format(leg_final_action[0], leg_final_action[1], leg_final_action[2], leg_final_action[3]))
		# print("ARM FINAL ACTION: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]\n".format(arm_final_action[0], arm_final_action[1], arm_final_action[2], arm_final_action[3]))
	DYN_CONFIG = {'lateralFriction_toe': 0.7, #1, # 0.6447185797960826, 
				  'lateralFriction_shank': 0.5, #0.6447185797960826  *  (0.351/0.512),
				  'contactStiffness': 2592, #2729, # 4173, #2157.4863390669952, 1615?
				  'contactDamping': 450, #414, #160, # 122, #32.46233575737161,    150?
				  'linearDamping': 0.03111169082496665, 
				  'angularDamping': 0.04396695866661371, 
				  'jointDamping': 0.03118494025640309, 
				  "max_force": 12, #10,
				  "mass_body": 9.5 #10.5
				  }

	LEG_ACTION = ["none", "parallel_offset", "hips_offset", "knees_offset", "hips_knees_offset"][1]

	env_args = {"render": True, "fix_body": True, "real_time": True, "immortal": False, 
				"version": 3, "normalised_abduct": True, "mode": "stand", "debug_tuner_enable" : True, "action_mode":"residual", "state_mode":"body_arm_leg_full_il", "leg_action_mode":LEG_ACTION,
				"tuner_enable": False, "action_tuner_enable": False, "A_range": (0.01, float("inf")), "gait": "triangle", "custom_dynamics": DYN_CONFIG, "param_opt": [0.022, 0, 0, 10, 0, 0, 0, 0, 0.011], #[0.022, 0, 0, 10, 0.1, 0.1, 0.1, 0.1, 0.011],
				"fast_error_update": True, "arm_pd_control": True, "custom_robot": {"k": 0.69}, "progressing": False, "max_steps": 2000,
				"leg_offset_multiplier": 0.01, "action_multiplier": 0.01, "sub_step_callback": sub_step_callback} # , "progressing_agent_factor": True

	dog = Dog(**env_args)
	# dog._set_dynamics(**DYN_CONFIG)

	n_epi = 0
	while True:
		done = False
		dog.reset()
		n_step = 0
		while not done:
			s, r, done, info =  dog.step([1]*6)
			n_step += 1
			print("LEG IDICATORS  ", dog.leg_state_indicators) 
		n_epi += 1


# simple_debug()