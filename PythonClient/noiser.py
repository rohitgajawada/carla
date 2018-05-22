import time
import random
import copy

class Noiser(object):



	# define frequency into noise events per minute
	# define the amount_of_time of setting the noise


	# NOISER GTA CONFIGURATION 
	# frequency=15, intensity = 8,min_noise_time_amount = 0.5

	# NOISER CARLA CONFIGURATION
	# frequency=15, intensity = 5 ,min_noise_time_amount = 0.5

	# NOISER CARLA AGENT CONFIGURATION
	# frequency=15, intensity = 5 ,min_noise_time_amount = 0.5

	# NOISER DEEPRC CONFIGURATION
	#frequency=20, intensity = 5 ,min_noise_time_amount = 0.5

	def __init__(self,noise_type,frequency=15,intensity = 30,min_noise_time_amount =0.5):



		self.noise_type = noise_type
		self.frequency = frequency
		self.noise_being_set = False
		self.noise_start_time =time.time()
		self.noise_end_time =time.time()+1
		self.min_noise_time_amount = min_noise_time_amount
		self.noise_time_amount =min_noise_time_amount + float(random.randint(50,150)/100.0)
		self.second_counter = time.time()
		self.steer_noise_time =0
		self.intensity =intensity
		self.remove_noise = False
		self.current_noise_mean  =0


	def set_noise(self):

		if self.noise_type == 'Spike' or 'Square':# spike noise there are no variations on current noise over time

			# flip between positive and negative
			coin = random.randint(0,1)
			if coin == 0: # negative
				self.current_noise_mean = 0.001#-random.gauss(0.05,0.02)
			else: # positive
				self.current_noise_mean = -0.001#random.gauss(0.05,0.02)

			

	def get_noise(self):

		if self.noise_type == 'Spike':
			if self.current_noise_mean >0:

				return min(0.55,self.current_noise_mean + (time.time() -self.noise_start_time)*0.03 *self.intensity)
			else:

				return max(-0.55,self.current_noise_mean - (time.time() -self.noise_start_time)*0.03*self.intensity)

		if self.noise_type == 'Square':
			if self.current_noise_mean >0:

				return min(1.0,self.current_noise_mean + (time.time() -self.noise_start_time)*0.03 *self.intensity)
			else:

				return max(-1.0,self.current_noise_mean - (time.time() -self.noise_start_time)*0.03*self.intensity)


	def get_noise_removing(self):
		#print 'REMOVING'
		added_noise = (self.noise_end_time -self.noise_start_time)*0.02 *self.intensity
		#print added_noise
		if self.noise_type == 'Spike':
			if self.current_noise_mean >0:
				added_noise = min(0.55,added_noise+self.current_noise_mean)
				return  added_noise - (time.time() -self.noise_end_time)*0.03*self.intensity
			else:
				added_noise = max(-0.55,self.current_noise_mean - added_noise)
				return added_noise + (time.time() -self.noise_end_time)*0.03*self.intensity


		if self.noise_type == 'Square':
			if self.current_noise_mean >0:
				added_noise = min(1.0,added_noise+self.current_noise_mean)
				return  added_noise - (time.time() -self.noise_end_time)*0.03*self.intensity
			else:
				added_noise = max(-1.0,self.current_noise_mean - added_noise)
				return added_noise + (time.time() -self.noise_end_time)*0.03*self.intensity



	def is_time_for_noise(self,steer):

		# Count Seconds
		second_passed=False
		if time.time() - self.second_counter >= 1.0:
			second_passed=True
			self.second_counter = time.time()


		if time.time() -self.noise_start_time >= self.noise_time_amount and not self.remove_noise and self.noise_being_set:
			self.noise_being_set=False
			self.remove_noise =True
			self.noise_end_time = time.time()
			

		if self.noise_being_set:
			return True


		if self.remove_noise:
			#print "TIME REMOVING ",(time.time()-self.noise_end_time)
			if (time.time()-self.noise_end_time) >(self.noise_time_amount): # if half the amount put passed
				self.remove_noise = False
				self.noise_time_amount = self.min_noise_time_amount + float(random.randint(50,200)/100.0)
				return False	
			else:
				return True
 
		if second_passed and not self.noise_being_set: # Ok, if noise is not being set but a second passed... we may start puting more

			seed = random.randint(0,60)
			if seed < self.frequency:
				if not self.noise_being_set:
					self.noise_being_set = True
					self.set_noise()
					self.steer_noise_time = steer
					self.noise_start_time = time.time()
				return True
			else:
				return False

		else:
			return False



	def set_noise_exist(self,noise_exist):
		self.noise_being_set = noise_exist



	def compute_noise(self,action,speed):

		#noisy_action = action
		if self.noise_type == 'None':
			return action,False,False



		if self.noise_type == 'Spike':

			if self.is_time_for_noise(action.steer):
				steer = action.steer

				if self.remove_noise:
					steer_noisy = max(min(steer + self.get_noise_removing()*(30/(1.5*speed+5)),1),-1)

				else:
					steer_noisy = max(min(steer+ self.get_noise()*(30/(1.5*speed+5)),1),-1)	
							

				noisy_action = copy.deepcopy(action)
				
				noisy_action.steer = steer_noisy

				#print 'timefornosie'
				return noisy_action,False,not self.remove_noise

			else:
				return action,False,False




		if self.noise_type == 'Square':

			if self.is_time_for_noise(action.steer):
				steer = action.steer

				'''if self.remove_noise:
					steer_noisy = max(min(steer + self.get_noise_removing(),1),-1)	#steer_noisy is the resultant of steer and noise

				else:
					steer_noisy = max(min(steer+ self.get_noise(),1),-1)	
							


				if steer_noisy > 0.5:
					steer_noisy = 1.0
				elif steer_noisy < -0.5:
					steer_noisy = -1.0
				else:
					steer_noisy = 0.0'''


				if self.remove_noise:
					noise_added = self.get_noise_removing()
				else:
					noise_added = self.get_noise()

				#print noise_added

				if noise_added > 0.35:
					noise_added = 1.0
				elif noise_added < -0.35:
					noise_added = -1.0
				else:
					noise_added = 0.0				

				#print noise_added
				steer_noisy = max(min(steer+ noise_added,1.0),-1.0)	
				#print steer_noisy			


				noisy_action = copy.deepcopy(action)
				
				noisy_action.steer = steer_noisy

				#print 'timefornosie'
				return noisy_action,False,not self.remove_noise
				
			else:
				return action,False,False


	
			
		if self.noise_type == 'Manual':
			if self.is_time_for_noise(action.steer):


				if self.remove_noise:
					drifting_time =self.noise_time_amount -(time.time() - self.noise_end_time)

				else:
					drifting_time =self.noise_time_amount -(time.time() - self.noise_start_time)

							


				return action,drifting_time,not self.remove_noise
			else:
				return action,0.0,False
				




if __name__ == '__main__':

	import matplotlib.pyplot as plt
	# Noise testing
	noise_input = []
	human_input = []
	noiser = Noiser('Spike')
	for i in range(500):
		human_action = Control()
		human_action.steer = 0.0
		human_action.gas = 0.0
		human_action.brake = 0.0
		human_action.hand_brake = 0.0
		human_action.reverse= 0.0


		human_input.append(human_action.steer)

		noisy_action,_,_ = noiser.compute_noise(human_action)
		time.sleep(0.01)
		noise_input.append(noisy_action.steer)

	plt.plot(range(500),human_input,'g',range(500),noise_input,'r')
	plt.show()

