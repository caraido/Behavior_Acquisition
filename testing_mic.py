import serial

with serial.Serial('COM3', 9600, timeout=2) as ser:
	result=ser.write(b'I_C_1:status_byte?')
	line1 = ser.readline()
	line2 = ser.readline()

	line1=line1.decode('ascii')
	line2=line2.decode('ascii')
	print(line1)
	print(line2)
