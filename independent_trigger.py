'''
import nidaqmx

TRIGGER_OUTPUT_CHANNEL = 'Dev1/ctr1'
TRIGGER_FREQUENCY= 15  # Hz
DUTY_CYCLE=0.5

trigger_task=nidaqmx.Task()
trigger_task.co_channels.add_co_pulse_chan_freq(
    TRIGGER_OUTPUT_CHANNEL,freq=TRIGGER_FREQUENCY,duty_cycle=DUTY_CYCLE
)

trigger_task.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
trigger_task.start()


'''
import time

import nidaqmx
from nidaqmx.constants import LineGrouping
from nidaqmx.stream_writers import DigitalSingleChannelWriter,CounterWriter
import numpy as np
import keyboard
import sys

#FREQUENCY=10 # Hz
#sleep_time=float(1/FREQUENCY) # in second
#DUTY_CYCLES=np.array([0.5],dtype='float')

sleep_time=2
CYCLE_LENGTH=2000
a = np.ones(CYCLE_LENGTH, dtype="uint32")
a[1000:CYCLE_LENGTH]=0
TRIGGER_OUTPUT_CHANNEL = "Dev1/port0/line0"

class optogenetics_LED:
    def __init__(self):
        self.task= nidaqmx.Task()
        self.task.do_channels.add_do_chan(TRIGGER_OUTPUT_CHANNEL, line_grouping=LineGrouping.CHAN_PER_LINE)
        self.writer=DigitalSingleChannelWriter(self.task.out_stream)
        self.running=False

    def run(self):
        while True:
            self.writer.write_many_sample_port_uint32(a)
            time.sleep(sleep_time)

    def end_run(self):
        self.task.stop()


def toggle_LED(writer:DigitalSingleChannelWriter,state:bool):
    if state:
        writer=None #start a new writer
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan("Dev1/port0/line0", line_grouping=LineGrouping.CHAN_PER_LINE)
            writer = DigitalSingleChannelWriter(task.out_stream, auto_start=True)
            while True:
                if keyboard.is_pressed('Esc'):
                    print('Exit')
                    sys.exit(0)

                writer.write_many_sample_port_uint32(a)
                time.sleep(sleep_time)
        print('LED on')
        return writer
    else:
        writer.



