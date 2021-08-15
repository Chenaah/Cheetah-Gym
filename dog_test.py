from dog import Dog

def test():

	dog = Dog(render=True, real_time=False, immortal=False, custom_dynamics=False, version=3, mode="stand")
	# dog.debug_dynamics()
	
	n_epi = 0
	while True:
		done = False

		dog.reset()
		print("OMEGA_PITCH: ", dog.get_full_state()[10])
		while not done:
		# p_set = p.readUserDebugParameter(p_debug)
		# y_set = p.readUserDebugParameter(y_debug)
		# r_set = p.readUserDebugParameter(r_debug)
		# print(f"set pitch to {p_set}")
		# p.resetBasePositionAndOrientation(dogId, [0, 0, 0.7], p.getQuaternionFromEuler([r_set,p_set,y_set]))
			s, r, done, info =  dog.step(dog.action_space.sample(), [0.6,0.6])
		n_epi += 1

		
def simple_test():
	global k_
	k_ = 0.6102
	dog = Dog(render=False, real_time=False, immortal=False, custom_dynamics=False, version=3, mode="stand")
	th2 = dog._theta1_hat(-1.00009)
	print(th2)

def test_pitch():
	dog = Dog(render=True, real_time=True, immortal=False, custom_dynamics=False, version=3, mode="stand")
	dog.reset()
	while True:
		# dog.step_jpos([-1.73918e-05, -1.46079, 0.99807, -1.8719e-05, -1.46079, 0.998076, 0.000439517, -1.09479, -1.00009, -0.000370475, -1.09484, -1.00013])
		dog.step_jpos()
		time.sleep(0.001)
		dog.get_full_state()
		print(dog.pitch)

def test_standup():
	dog = Dog(render=True, real_time=True, immortal=False, custom_dynamics=False, version=3, mode="standup")
	dog.reset()
	while True:
		pass

def test_observation():
	DYN_CONFIG = {#'mass_body': 5.555,
				  'lateralFriction_toe': 1, # 0.6447185797960826, 
				  'lateralFriction_shank': 0.737, #0.6447185797960826  *  (0.351/0.512),
				  'contactStiffness': 4173, #2157.4863390669952, 
				  'contactDamping': 122, #32.46233575737161, 
				  'linearDamping': 0.03111169082496665, 
				  'angularDamping': 0.04396695866661371, 
				  'jointDamping': 0.03118494025640309, 
				  # 'w_y_offset': 0.0021590823485152914
				  }


	dog = Dog(render=True, real_time=False, immortal=False, fix_body=True, custom_dynamics=DYN_CONFIG, version=3, mode="standup", num_history_observation=9, state_mode="body_arm_leg_full", 
			  action_mode="residual", leg_action_mode="none", randomise=True)
	# dog.set_dynamics(**DYN_CONFIG)
	while True:
		done = False
		s = dog.reset()
		assert len(list(s)) == 14*10
		while not done:
			s, r, done, info =  dog.step([0,0,0,0,0,0], param_opt=[0.086, 0, 0, 5.368, 0.1, 0.1, 0.1, 0.1, 0.11])
			# dog.step_jpos([0]*12)
			assert len(list(s)) == 14*10