import re
import datajoint as dj
import numpy
import numpy as np
import pandas as pd
import json
import datetime
from global_settings import config, N_ANIMALS_TO_DISPLAY


def get_notes(updates):
	if len(updates[4])!=0:
		windowA=f"{updates[4]} {updates[3]} in A, "
	else:
		windowA=f"{updates[3]} in A,"

	if len(updates[6])!=0:
		windowB=f"{updates[6]} {updates[5]} in B, "
	else:
		windowB=f"{updates[5]} in B,"

	if len(updates[8])!=0:
		windowC=f"{updates[8]} {updates[7]} in C, "
	else:
		windowC=f"{updates[7]} in C, "

	text=windowA+windowB+windowC+updates[10]
	return text

class DjConn:

	def __init__(self, config=config):
		self.config=config
		self.connection=None
		self.is_connected=False
		self.sl_test=None
		self.sl=None
		self.sl_behavior=None
		self.sln_animal = None

	def connect_to_datajoint(self):
		if self.is_connected:
			return True
			
		for key, value in self.config.items():
			dj.config[key] = value
		self.connection=dj.conn()
		self.is_connected=self.connection.is_connected
		if self.is_connected:
			for schema in dj.list_schemas():
				setattr(self, schema, dj.create_virtual_module(f'{schema}.py', schema))
				# self.sl = dj.create_virtual_module('sl.py', 'sl')
				# self.sl.Animal() == `sl`.`animal`
				# self.sln_animal = dj.create_virtual_module('sln_animal.py','sln_animal')

		return self.is_connected

	def drop_connection(self):
		if self.connection is not None:
			self.connection.close()

	# def get_test_schema(self):
	# 	if self.connection:
	# 		self.sl_test = dj.create_virtual_module('sl_test.py', 'sl_test')
	# 		return True
	# 	else:
	# 		return False

	# def get_main_schema(self):
	# 	if self.connection:
	# 		self.sl = dj.create_virtual_module('sl.py', 'sl')
	# 		return True
	# 	else:
	# 		return False

	# def get_behavior_schema(self):
	# 	if self.connection:
	# 		self.sl_behavior = dj.create_virtual_module('sl_behavior.py', 'sl_behavior')
	# 		return True
	# 	else:
	# 		return False

		# fetch entities from tables
	def get_all_AnimalType(self):
		# animal_types,types_description = sl.TestAnimalType.fetch()
		all_animal_types= self.sl_behavior.TestAnimalType.fetch()
		animal_types=[i[0] for i in all_animal_types]
		types_description=[i[1] for i in all_animal_types]

		return animal_types

	# fetch entities from tables
	def get_AnimalIds(self, count=5):
		if self.sln_animal is not None:
			all_ID=(self.sln_animal.AnimalEvent * self.sln_animal.SocialBehaviorSession).fetch('animal_id', order_by=('date desc', 'time desc', 'entry_time desc'), limit=2*count)
			unique_ID, ind=np.unique(all_ID, return_index=True)
			return unique_ID[np.argsort(ind)][:count].tolist()

	def get_StimAnimalIds(self, count=None):
		raise 'Not yet implemented'
		if self.sln_animal is not None:
			# all_ID=self.sl.AnimalEventSocialBehaviorSession.fetch('animal_id', order_by=('date desc', 'time desc', 'entry_time desc'), limit=count)
			unique_ID, ind=np.unique(all_ID, return_index=True)
			return unique_ID[np.argsort(ind)][:count].tolist()

	# fetch all behavior session
	def get_all_Sessions(self):
		if self.sln_animal is not None:
			all_Sessions=pd.DataFrame(self.sln_animal.SocialBehaviorSession.fetch())
			return all_Sessions
		else:
			return


	# fetch entities from tables
	def get_all_StimulusType(self):
		if self.sln_animal is not None:
			all_stimulus_type=self.sl_behavior.VisualStimulusType.fetch()
			stimulus_type=[i[0] for i in all_stimulus_type]
			print(stimulus_type)
			return stimulus_type
		else:
			print('cannot find schema')
			return

	# fetch entities from tables
	def get_all_ExperimentType(self):
		if self.sln_animal is not None:
			all_behavior_experiment_type=self.sl_behavior.SocialBehaviorExperimentType.fetch()
			behavior_experiment_type=[i[0] for i in all_behavior_experiment_type]
			print(behavior_experiment_type)
			return behavior_experiment_type
		else:
			print('cannot find schema')
			return

	# def update_json(self):
	# 	file={}
	# 	animalID=self.get_AnimalIds(count=N_ANIMALS_TO_DISPLAY)
	# 	animalType=self.get_all_AnimalType()
	# 	stimulusType=self.get_all_StimulusType()
	# 	experimentType=self.get_all_ExperimentType()
	# 	file['animalID']=[str(a) for a in animalID]
	# 	file['animalType']=[str(a) for a in animalType]
	# 	file['experimentType']=[str(a) for a in experimentType]
	# 	file['windows'] = [str(a) for a in stimulusType]

	# 	with open(namespace_path,'w') as f:
	# 		json.dump(file,f)

	# 	return # something tells the gui to update?

	def package_data(self):
		animalIds = self.get_AnimalIds(count=N_ANIMALS_TO_DISPLAY)
		return {
			'recent_test_animals': animalIds,
			'recent_stim_animals': animalIds, #TODO: implement get_StimAnimalIds
			'animal_types': self.get_all_AnimalType(),
			'stimulus_types': self.get_all_StimulusType(),
			'experiment_types': self.get_all_ExperimentType()
		}

	def find_animalID(self, animal_IDs):
		#checks for animal IDs in sl.Animal
		#returns false if animal id is missing
		#excepts the case that no animal_ID is provided (returns true)

		all_IDs=((self.sln_animal.Animal & [f"animal_id={animal_ID}" for animal_ID in animal_IDs if animal_ID is not '' ])- (self.sln_animal.Deceased * self.sln_animal.AnimalEvent)).fetch('animal_id')
		is_in_DB = [animal_ID is '' or int(animal_ID) in all_IDs for animal_ID in animal_IDs]
		return is_in_DB

	def update_session(self,some_update):
		animal_id=some_update[0]
		if self.find_animalID(animal_id):
			user_name=some_update[9]
			purpose=some_update[2]
			animal_type_name=some_update[1]
			notes=get_notes(some_update)

			today=datetime.datetime.today()
			date_now=today.date()
			time_now=today.time()
			thistime=datetime.timedelta(days=0,hours=time_now.hour,minutes=time_now.minute,seconds=time_now.second)

			event_insert=dict(animal_id=animal_id,
							 user_name=user_name,
							 date=date_now,
							 time=thistime,
							 notes=notes)
			session_insert=dict(
							 event_id=None,
							 purpose=purpose,
							 animal_type_name=animal_type_name,
							 recorded='T')

			try:
				with self.connection.transaction:
					self.sln_animal.AnimalEvent.insert1(event_insert)
					event_id=self.get_matching_event_id(event_insert)
					if event_id is None: #does this ever happen???
						raise

					session_insert['event_id'] = event_id
					self.sln_animal.SocialBehaviorSession.insert1(session_insert)
					# event_id=self.get_latest_event_id()
					items = []
					for stim_type,stim_ID,arm in zip(some_update[3:8:2],some_update[4:9:2], ('A','B','C')):
						items.append(dict(
								event_id = event_id,
								arm = arm,
								stim_type=stim_type,
								stimulus_animal_id = stim_ID if stim_ID != '' else None
							))
					self.sln_animal.SocialBehaviorSession.Stimulus.insert(items)
				info='session insertion successful. Start recording'
			except:
				event_id=False
				info='session insertion unsuccessful. Abort recording'

			return event_id,info
		else:
			return False,'Test animal ID does not match the database! \n Session is not inserted on Datajoint. Still Recording'
	
	def get_matching_event_id(self, event):
		event['date'] = str(event['date'])
		event['time'] = str(event['time'])
		
		return (self.sln_animal.AnimalEvent & event).fetch1('event_id')

	def get_latest_event_id(self):
		#deprecated: see below

		#TODO: ought'nt we fetch the entry that matches ours, in case of collisions?
		# if we're in a transaction, it's probably okay
		# also we don't expect any collisions in normal use

		sessions=self.get_all_Sessions() #we don't need all of this!!
		event_ids=sessions['event_id']
		latest_event_id=numpy.array(event_ids)[-1]
		return latest_event_id

	# talk to dj for permission
	def add_new_type_to_dj(self,some_update):

		#no clue what this block is for... what is the stuff in parens that we're ignoring?:
		some_update[3] = re.split(r'[(](\d+)[)]',some_update[3])[0]
		some_update[5] = re.split(r'[(](\d+)[)]', some_update[5])[0]
		some_update[7] = re.split(r'[(](\d+)[)]', some_update[7])[0]

		# animalID = [str(a) for a in self.get_AnimalIds()]
		animalType = self.get_all_AnimalType()
		stimulusType = self.get_all_StimulusType()
		experimentType = self.get_all_ExperimentType()
		if str(some_update[0]) == 'unnamedAnimal':
			return True,'unnamed Animal. Datajoint not updated'
		# is_avail_windowA_DJID = some_update[4] in animalID or len(some_update[4])== 0
		# is_avail_windowB_DJID = some_update[6] in animalID or len(some_update[6])== 0
		# is_avail_windowC_DJID = some_update[8] in animalID or len(some_update[8])== 0
		a_in_DJ,b_in_DJ,c_in_DJ = self.find_animalID(some_update[4:9:2])

		su_ids = [some_update[i]  for i in [0,4,6,8] if some_update[i]!='']
		if len(set(su_ids)) != len(su_ids):
			return False,'Tried to enter the same mouse in multiple windows!'

		if not a_in_DJ:
			# database must have the DJID of this mouse first
			return False,f"can't find this mouse DJID: {some_update[4]} in database. Datajoint not updated"
		if not b_in_DJ:
			# database must have the DJID of this mouse first
			return False,f"can't find this mouse DJID: {some_update[6]} in database. Datajoint not updated"
		if not c_in_DJ:
			# database must have the DJID of this mouse first
			return False,f"can't find this mouse DJID: {some_update[7]} in database. Datajoint not updated"

		try:
			#TODO: no need to query the stimulus types again... just bring this down from .getAllStimulusTypes
			tested = []

			# TODO: no need to do so many insert1's... insert as a batch
			#honestly we should just get rid of this maybe...
			count=0
			if some_update[1] not in animalType:
				self.sl_behavior.TestAnimalType.insert1([str(some_update[1]),''])
				count+=1
			if some_update[2] not in experimentType:
				self.sl_behavior.SocialBehaviorExperimentType.insert1([str(some_update[2]),''])
				count+=1
			if some_update[3] not in stimulusType:
				if len(some_update[4])==0:
					needs_id='F'
					tested.append(False)
				else:
					needs_id='T'
					tested.append(True)

				self.sl_behavior.VisualStimulusType.insert1([some_update[3],needs_id,''])
				count += 1
			else:
				tested.append((self.sl_behavior.VisualStimulusType & f'stim_type="{some_update[3]}"').fetch1('needs_id') ==  'T')
			if some_update[5] not in stimulusType:
				if len(some_update[6])==0:
					needs_id='F'
					tested.append(False)
				else:
					needs_id='T'
					tested.append(True)
				self.sl_behavior.VisualStimulusType.insert1([some_update[5],needs_id,''])
				count += 1
			elif some_update[5]==some_update[3]:
				tested.append(tested[0])
			else:
				tested.append((self.sl_behavior.VisualStimulusType & f'stim_type="{some_update[5]}"').fetch1('needs_id') ==  'T')
			
			if some_update[7] not in stimulusType:
				if len(some_update[7])==0:
					needs_id='F'
					tested.append(False)
				else:
					needs_id='T'
					tested.append(True)
				self.sl_behavior.VisualStimulusType.insert1([some_update[7],needs_id,''])
				count += 1
			elif some_update[7]==some_update[3]:
				tested.append(tested[0])
			elif some_update[7]==some_update[5]:
				tested.append(tested[1])
			else:
				tested.append((self.sl_behavior.VisualStimulusType & f'stim_type="{some_update[7]}"').fetch1('needs_id') ==  'T')
			
			if count:
				message='Updated the type!'
			else:
				message=''

			if tested != [some_update[4]!='',some_update[6]!='',some_update[8]!='']:
				return False,'The specified stimulus types require DJIDs!'

			return True,message
		except:
			return False,'Failed to update the type '
		# with c.transaction(): # try to update the entire package
			# status = sl.testanimaltype.insert1({})
			# status =sl.testanimaltype.insert1({})
			# status =sl.testanimaltype.insert1({})
			#for arm in rig:
			#	sl.AnimalEventSocialBehaviorSessionStimulus.insert1({...})


if __name__ == '__main__':
	conn=DjConn()
	connected=conn.connect_to_datajoint()
	print(connected)
	# status=conn.get_main_schema()
	print(status)
	# conn.update_json()

	pass

