import numpy as np
import nidaqmx
from nidaqmx.constants import LineGrouping
from nidaqmx.stream_writers import DigitalSingleChannelWriter
from AcquisitionObject import AcquisitionObject
import time

FREQUENCY= 200 # Hz
DUTY_CYCLES=0.1
sleep_time=float(1/FREQUENCY) # in second
channel="Dev1/port0/line0"
line_grouping=LineGrouping.CHAN_PER_LINE

class LED:

	def __init__(self):
		self.led_task=nidaqmx.Task()
		self.led_task.do_channels.add_do_chan(channel,line_grouping=line_grouping)
		self.writer=DigitalSingleChannelWriter(self.led_task.out_stream)

		self._running=False

	def start(self):
		self.a=np.zeros(2,dtype='unit32')
		self.a[0]=1
		self.led_task.start()

	def run(self):
		self._running=True
		while self._running:
			self.writer.write_many_sample_port_uint32(self.a)
			time.sleep=sleep_time

	def stop(self):
		self._running=False
		self.led_task.stop()


