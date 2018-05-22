#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.carla_server_pb2 import Control
from carla import image_converter

from human_control import HumanDriver
from noiser import Noiser

import configparser
import h5py
from skimage.transform import resize
import time
import datetime
from PIL import Image
import numpy as np
import os
import math as m


class Recorder(object):
	def __init__(self,file_prefix,resolution=[800,600],current_file_number=0,\
		record_image=True,image_cut =[0,600],camera_dict=[]):

		self._number_of_seg_classes = 13
		self._record_image_hdf5 = True
		self._number_images_per_file = 100
		self._file_prefix = file_prefix
		self._image_size2 =resolution[1]
		self._image_size1 = resolution[0]
		self._number_rewards = 28
		self._image_cut = image_cut
		self._camera_dict = camera_dict
		if not os.path.exists(self._file_prefix):
			os.mkdir(self._file_prefix)


		self._current_file_number = current_file_number
		self._current_hf = self._create_new_db()

		self._csv_writing_file = self._file_prefix + 'outputs.csv'
		self._current_pos_on_file =0


	def _create_new_db(self):

		hf = h5py.File( self._file_prefix +'data_'+ str(self._current_file_number).zfill(5) +'.h5', 'w')
		# print(self._image_size2, self._image_size1)
		self.data_center= hf.create_dataset('rgb', (self._number_images_per_file,self._image_size2,self._image_size1,3),dtype=np.float64)
		self.segs_center= hf.create_dataset('labels', (self._number_images_per_file,self._image_size2,self._image_size1,1),dtype=np.uint8)
		self.depth_center= hf.create_dataset('depth', (self._number_images_per_file,self._image_size2,self._image_size1,3),dtype=np.uint8)


		self.data_rewards  = hf.create_dataset('targets', (self._number_images_per_file, self._number_rewards),'f')
		return hf

	def record(self,measurements,sensor_data,action,action_noise,direction):
		data = [measurements,sensor_data,action,action_noise,direction]
		self._write_to_disk(data)

	def _write_to_disk(self,data):
		# Use the dictionary for this

		measurements = data[0]

		sensor_data = data[1]
		actions = data[2]
		action_noise = data[3]
		direction = data[4]
		# waypoints = data[5]
		sensor_list = sensor_data.keys()

		for i, camera in enumerate(self._camera_dict):
			if self._current_pos_on_file == self._number_images_per_file:
				self._current_file_number += 1
				self._current_pos_on_file = 0
				self._current_hf.close()
				self._current_hf = self._create_new_db()

			pos = self._current_pos_on_file
			capture_time   = int(round(time.time() * 1000))
			image = image_converter.to_bgra_array(sensor_data[camera['SceneFinal']])
			image = image[300:600,:,:3]
			image = image[:,:,:3]
			image = image[:, :, ::-1]
			image = resize(image,[self._image_size2,self._image_size1])
			# print (type(image[0][0][0]))
			self.data_center[pos] = image

			scene_seg = image_converter.to_bgra_array(sensor_data[camera['SemanticSegmentation']])[self._image_cut[0]:self._image_cut[1],:,2]
			scene_seg = resize(scene_seg,[self._image_size2,self._image_size1])
			scene_seg = scene_seg[:,:,np.newaxis]
			self.segs_center[pos] = scene_seg

			depth = image_converter.to_bgra_array(sensor_data[camera['Depth']])[self._image_cut[0]:self._image_cut[1],:,:3]
			depth = resize(depth,[self._image_size2,self._image_size1])
			self.depth_center[pos] = depth
			self.data_rewards[pos,0]  = actions.steer
			self.data_rewards[pos,1]  = actions.throttle
			self.data_rewards[pos,2]  = actions.brake
			self.data_rewards[pos,3]  = actions.hand_brake
			self.data_rewards[pos,4]  = actions.reverse
			self.data_rewards[pos,5]  = action_noise.steer
			# self.data_rewards[pos,5] = 0
			self.data_rewards[pos,6]  = action_noise.throttle
			# self.data_rewards[pos,6] = 0
			self.data_rewards[pos,7]  = action_noise.brake
			# self.data_rewards[pos,7] = 0
			self.data_rewards[pos,8]  = measurements.player_measurements.transform.location.x
			self.data_rewards[pos,9]  = measurements.player_measurements.transform.location.y
			self.data_rewards[pos,10]  = measurements.player_measurements.forward_speed
			self.data_rewards[pos,11]  = measurements.player_measurements.collision_other
			self.data_rewards[pos,12]  = measurements.player_measurements.collision_pedestrians
			self.data_rewards[pos,13]  = measurements.player_measurements.collision_vehicles
			self.data_rewards[pos,14]  = measurements.player_measurements.intersection_otherlane
			self.data_rewards[pos,15]  = measurements.player_measurements.intersection_offroad
			self.data_rewards[pos,16]  = measurements.player_measurements.acceleration.x
			self.data_rewards[pos,17]  = measurements.player_measurements.acceleration.y
			self.data_rewards[pos,18]  = measurements.player_measurements.acceleration.z
			self.data_rewards[pos,19]  = 0
			self.data_rewards[pos,20]  = 0
			self.data_rewards[pos,21]  = measurements.player_measurements.transform.orientation.x
			self.data_rewards[pos,22]  = measurements.player_measurements.transform.orientation.y
			self.data_rewards[pos,23]  = measurements.player_measurements.transform.orientation.z
			self.data_rewards[pos,24]  = direction
			self.data_rewards[pos,25]  = i
			# self.data_rewards[pos,26]  = float(self._camera_dict[i][1])
			self.data_rewards[pos,26] = 0.0

			self._current_pos_on_file += 1

	def close(self):

		self._current_hf.close()

def get_camera_dict(ini_file, cam_nums):
	cams = ['Camera' + cam for cam in cam_nums.split(',')]
	config = configparser.ConfigParser()
	config.read(ini_file)
	cameras = []
	# sensors =  config['CARLA/Sensor']['Sensors']
	# sensors = sensors.split(',')
	for cam in cams:
		scene_cam = cam
		seg_cam = cam + "/SemanticSegmentation"
		depth_cam = cam + "/Depth"
		camera = {'SceneFinal': scene_cam, 'SemanticSegmentation': seg_cam, 'Depth': depth_cam}
		cameras.append(camera)
	return cameras

def run_carla_client(args):
	# Here we will run 3 episodes with 300 frames each.
	number_of_episodes = 3
	frames_per_episode = 5000

	# We assume the CARLA server is already waiting for a client to connect at
	# host:port. To create a connection we can use the `make_carla_client`
	# context manager, it creates a CARLA client object and starts the
	# connection. It will throw an exception if something goes wrong. The
	# context manager makes sure the connection is always cleaned up on exit.
	if args.humandriver:
		driver = HumanDriver()

	with make_carla_client(args.host, args.port) as client:
		print('CarlaClient connected')
		if args.record:
			folder_name = str(datetime.datetime.today().day) + '_' + 'Carla_' + '_' + args.experiment_name
			camera_dict = get_camera_dict(args.settings_filepath, args.cameras)
			res = [int(i) for i in args.resolution.split(',')]
			image_cut = [int(i) for i in args.image_cut.split(',')]
			print (res, image_cut)
			recorder = Recorder(args.path_to_save + folder_name + '/', res,\
			image_cut=image_cut,camera_dict=camera_dict)
		for episode in range(0, number_of_episodes):
			# Start a new episode.
			if args.settings_filepath is None:
				# Create a CarlaSettings object. This object is a wrapper around
				# the CarlaSettings.ini file. Here we set the configuration we
				# want for the new episode.
				settings = CarlaSettings()
				settings.set(
					SynchronousMode=True,
					SendNonPlayerAgentsInfo=True,
					NumberOfVehicles=0,
					NumberOfPedestrians=4,
					WeatherId=random.choice([1, 3, 7, 8, 14]),
					QualityLevel=args.quality_level)
				settings.randomize_seeds()

				# Now we want to add a couple of cameras to the player vehicle.
				# We will collect the images produced by these cameras every
				# frame.

				# The default camera captures RGB images of the scene.
				camera0 = Camera('CameraRGB')
				# Set image resolution in pixels.
				camera0.set_image_size(800, 600)
				# Set its position relative to the car in meters.
				camera0.set_position(0.30, 0, 1.30)
				settings.add_sensor(camera0)

				# Let's add another camera producing ground-truth depth.
				camera1 = Camera('CameraDepth', PostProcessing='Depth')
				camera1.set_image_size(800, 600)
				camera1.set_position(0.30, 0, 1.30)
				settings.add_sensor(camera1)

				if args.lidar:
					lidar = Lidar('Lidar32')
					lidar.set_position(0, 0, 2.50)
					lidar.set_rotation(0, 0, 0)
					lidar.set(
						Channels=32,
						Range=50,
						PointsPerSecond=100000,
						RotationFrequency=10,
						UpperFovLimit=10,
						LowerFovLimit=-30)
					settings.add_sensor(lidar)

			else:

				# Alternatively, we can load these settings from a file.
				with open(args.settings_filepath, 'r') as fp:
					settings = fp.read()
					print (settings)

			# Now we load these settings into the server. The server replies
			# with a scene description containing the available start spots for
			# the player. Here we can provide a CarlaSettings object or a
			# CarlaSettings.ini file as string.
			scene = client.load_settings(settings)

			# Choose one player start at random.
			number_of_player_starts = len(scene.player_start_spots)
			player_start = random.randint(0, max(0, number_of_player_starts - 1))

			# Notify the server that we want to start the episode at the
			# player_start index. This function blocks until the server is ready
			# to start the episode.
			print('Starting new episode...')
			client.start_episode(player_start)

			# Iterate every frame in the episode.
			ct = 0
			for frame in range(0, frames_per_episode):

				# Read the data produced by the server this frame.
				measurements, sensor_data = client.read_data()
				# print(measurements, sensor_data)
				im = Image.frombytes(mode='RGBA', size=(200, 88), data=sensor_data['Camera0'].raw_data, decoder_name="raw")
				im.save('test/test.png')

				# print (type(measurements))
				# Print some of the measurements.
				# print_measurements(measurements)

				# Save the images to disk if requested.
				if args.save_images_to_disk:
					for name, measurement in sensor_data.items():
						filename = args.out_filename_format.format(episode, name, frame)
						measurement.save_to_disk(filename)

				# We can access the encodnearested data of a given image as numpy
				# array using its "data" property. For instance, to get the
				# depth value (normalized) at pixel X, Y
				#
				#     depth_array = sensor_data['CameraDepth'].data
				#     value_at_pixel = depth_array[Y, X]
				#
				# Now we have to send the instructions to control the vehicle.
				# If we are in synchronous mode the server will pause the
				# simulation until we send this control.
				control = Control()
				if args.humandriver:
					control = driver.computeControl()
					client.send_control(control)

				elif args.autopilot:
					# Together with the measurements, the server has sent the
					# control that the in-game autopilot would do this frame. We
					# can enable autopilot by sending back this control to the
					# server. We can modify it if wanted, here for instance we
					# will add some noise to the steer.
					control = measurements.player_measurements.autopilot_control
					control.steer += random.uniform(-0.1, 0.1)
					client.send_control(control)

				else:
					control.steer = random.uniform(-1.0, 1.0)
					control.throttle=0.5
					control.brake=0.0
					control.hand_brake=False
					control.reverse=False
					client.send_control(control)

				if args.record:
					noiser = Noiser(noise_type='Spike', frequency=15, intensity = 5 ,min_noise_time_amount = 0.5)
					action_noisy,drifting_time,will_drift = noiser.compute_noise(actions,speed)
					direction = 2
					actions = control
					if measurements.player_measurements.forward_speed <=0.1:
						print ("No move Frame no: ", frame)
					else:
						print ("Yes!! move Frame no: ", frame)
					recorder.record(measurements,sensor_data,actions,action_noisy,direction)

def print_measurements(measurements):
	number_of_agents = len(measurements.non_player_agents)
	player_measurements = measurements.player_measurements
	message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
	message += '{speed:.0f} km/h, '
	message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
	message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
	message += '({agents_num:d} non-player agents in the scene)'
	message = message.format(
		pos_x=player_measurements.transform.location.x,
		pos_y=player_measurements.transform.location.y,
		speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
		col_cars=player_measurements.collision_vehicles,
		col_ped=player_measurements.collision_pedestrians,
		col_other=player_measurements.collision_other,
		other_lane=100 * player_measurements.intersection_otherlane,
		offroad=100 * player_measurements.intersection_offroad,
		agents_num=number_of_agents)
	print_over_same_line(message)


def main():
	argparser = argparse.ArgumentParser(description=__doc__)
	argparser.add_argument(
		'-v', '--verbose',
		action='store_true',
		dest='debug',
		help='print debug information')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='localhost',
		help='IP of the host server (default: localhost)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'-a', '--autopilot',
		action='store_true',
		help='enable autopilot')
	argparser.add_argument(
		'-l', '--lidar',
		action='store_true',
		help='enable Lidar')
	argparser.add_argument(
		'-q', '--quality-level',
		choices=['Low', 'Epic'],
		type=lambda s: s.title(),
		default='Epic',
		help='graphics quality level, a lower level makes the simulation run considerably faster.')
	argparser.add_argument(
		'-i', '--images-to-disk',
		action='store_true',
		dest='save_images_to_disk',
		help='save images (and Lidar data if active) to disk')
	argparser.add_argument(
		'-c', '--carla-settings',
		metavar='PATH',
		dest='settings_filepath',
		default=None,
		help='Path to a "CarlaSettings.ini" file')
	argparser.add_argument(
		'-path', '--path-to-save',
		dest='path_to_save',
		default='./_out',
		type=str,
		help='Path to a where the data is saved')
	argparser.add_argument(
		'-reso', '--resolution',
		default='200, 88',
		type=str,
		help='Resolution of saved images')
	argparser.add_argument(
		'-exp', '--experiment-name',
		default='exp1',
		dest='experiment_name',
		type=str,
		help='Name of the experiment')
	argparser.add_argument(
		'-ic', '--image-cut',
		dest='image_cut',
		default='300, 600',
		type=str,
		help='Path to a where the data is saved')
	argparser.add_argument(
		'-rec', '--record',
		action='store_true',
		help='Whether to record to disk as hd5f files')
	argparser.add_argument(
		'-cam', '--cameras',
		default='0,15,30,N15,N30',
		type=str,
		help='Cameras used')
	argparser.add_argument(
		'-hd', '--humandriver',
		action='store_true',
		help='enable human driver')

	args = argparser.parse_args()

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
	logging.info('listening to server %s:%s', args.host, args.port)
	args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

	while True:
		try:
			run_carla_client(args)
			print('Done.')
			return

		except TCPConnectionError as error:
			logging.error(error)
			time.sleep(1)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')
