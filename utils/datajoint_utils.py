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
		all_animal_types= self.sl.TestAnimalType.fetch()
		animal_types=[i[0] for i in all_animal_types]
		types_description=[i[1] for i in all_animal_types]

		return animal_types

	# fetch entities from tables
	def get_AnimalIds(self, count=5):
		if self.sl is not None:
			all_ID=self.sl.AnimalEventSocialBehaviorSession.fetch('animal_id', order_by=('date desc', 'time desc', 'entry_time desc'), limit=2*count)
			unique_ID, ind=np.unique(all_ID, return_index=True)
			return unique_ID[np.argsort(ind)][:count].tolist()

	def get_StimAnimalIds(self, count=None):
		raise 'Not yet implemented'
		if self.sl is not None:
			# all_ID=self.sl.AnimalEventSocialBehaviorSession.fetch('animal_id', order_by=('date desc', 'time desc', 'entry_time desc'), limit=count)
			unique_ID, ind=np.unique(all_ID, return_index=True)
			return unique_ID[np.argsort(ind)][:count].tolist()

	# fetch all behavior session
	def get_all_Sessions(self):
		if self.sl is not None:
			all_Sessions=pd.DataFrame(self.sl.AnimalEventSocialBehaviorSession.fetch())
			return all_Sessions
		else:
			return


	# fetch entities from tables
	def get_all_StimulusType(self):
		if self.sl is not None:
			all_stimulus_type=self.sl.BehaviorVisualStimulusType.fetch()
			stimulus_type=[i[0] for i in all_stimulus_type]
			print(stimulus_type)
			return stimulus_type
		else:
			print('cannot find schema')
			return

	# fetch entities from tables
	def get_all_ExperimentType(self):
		if self.sl is not None:
			all_behavior_experiment_type=self.sl.SocialBehaviorExperimentType.fetch()
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

	def update_session(self,some_update):
		animal_id=some_update[0]
		all_ids=[str(a) for a in self.get_AnimalIds()]
		if animal_id in all_ids:
			user_name=some_update[9]
			purpose=some_update[2]
			animal_type_name=some_update[1]
			notes=get_notes(some_update)

			today=datetime.datetime.today()
			date_now=today.date()
			time_now=today.time()
			thistime=datetime.timedelta(days=0,hours=time_now.hour,minutes=time_now.minute,seconds=time_now.second)

			insert_item=dict(animal_id=animal_id,
							 user_name=user_name,
							 purpose=purpose,
							 animal_type_name=animal_type_name,
							 date=date_now,
							 time=thistime,
							 recorded='T',
							 notes=notes)

			try:
				self.sl.AnimalEventSocialBehaviorSession.insert1(insert_item)
				event_id=self.get_latest_event_id()
				info='session insertion successful. Start recording'
			except:
				event_id=False
				info='session insertion unsuccessful. Abort recording'

			return event_id,info
		else:
			return 'None','Test animal ID does not match the database! \n Session is not inserted on Datajoint. Still Recording'

	def get_latest_event_id(self):
		sessions=self.get_all_Sessions()
		event_ids=sessions['event_id']
		latest_event_id=numpy.array(event_ids)[-1]
		return latest_event_id

	# talk to dj for permission
	def add_new_type_to_dj(self,some_update):
		windowA=re.split(r'[(](\d+)[)]',some_update[3])
		windowB = re.split(r'[(](\d+)[)]', some_update[5])
		windowC = re.split(r'[(](\d+)[)]', some_update[7])

		animalID = [str(a) for a in self.get_AnimalIds()]
		animalType = self.get_all_AnimalType()
		stimulusType = self.get_all_StimulusType()
		experimentType = self.get_all_ExperimentType()
		if str(some_update[0]) == 'unnamedAnimal':
			return True,'unnamed Animal. Datajoint not updated'
		is_avail_windowA_DJID = some_update[4] in animalID or len(some_update[4])== 0
		is_avail_windowB_DJID = some_update[6] in animalID or len(some_update[6])== 0
		is_avail_windowC_DJID = some_update[8] in animalID or len(some_update[8])== 0

		if not is_avail_windowA_DJID:
			# database must have the DJID of this mouse first
			return False,f"can't find this mouse DJID: {some_update[4]} in database. Datajoint not updated"
		if not is_avail_windowB_DJID:
			# database must have the DJID of this mouse first
			return False,f"can't find this mouse DJID: {some_update[6]} in database. Datajoint not updated"
		if not is_avail_windowC_DJID:
			# database must have the DJID of this mouse first
			return False,f"can't find this mouse DJID: {some_update[7]} in database. Datajoint not updated"

		try:
			count=0
			if some_update[1] not in animalType:
				self.sl.TestAnimalType.insert1([str(some_update[1]),''])
				count+=1
			if some_update[2] not in experimentType:
				self.sl.SocialBehaviorExperimentType.insert1([str(some_update[2]),''])
				count+=1
			if windowA[0] not in stimulusType:
				if len(some_update[4])==0:
					needs_id='F'
				else:
					needs_id='T'
				self.sl.BehaviorVisualStimulusType.insert1([windowA[0],needs_id,''])
				count += 1
			if windowB[0] not in stimulusType:
				if len(some_update[6])==0:
					needs_id='F'
				else:
					needs_id='T'
				self.sl.BehaviorVisualStimulusType.insert1([windowB[0],needs_id,''])
				count += 1
			if windowC[0] not in stimulusType:
				if len(some_update[7])==0:
					needs_id='F'
				else:
					needs_id='T'
				self.sl.BehaviorVisualStimulusType.insert1([windowC[0],needs_id,''])
				count += 1

			if count:
				message='Updated the type!'
			else:
				message=''
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

