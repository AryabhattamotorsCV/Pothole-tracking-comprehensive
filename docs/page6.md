# Code for sensors used in testing

In this section I will be explaining code used to test out different sensors, even though these sensor's work with ARDUINO code, there are ways to make it interact with python code using frameworks like pyserial and pyfirmata.

# Accelerometer code

```py title="Accelerometer_position_tracking.py" linenums="1"
import serial
import time

# Serial setup
ser = serial.Serial('COM7', 115200, timeout=1)
time.sleep(2)

# Motion Variables
velocity_x, velocity_y, velocity_z = 0, 0, 0
distance_x, distance_y, distance_z = 0, 0, 0
last_time = time.time()

ACCEL_THRESHOLD = 0.5  # Ignore small accelerations since the sensor has a lot of noise
DECAY_FACTOR = 0.98    # This constant helps in controlling the increment of velocity, if this is not done velocity will keep on increasing 

while True:
    try:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            if data:
                # Parse received data: yaw, pitch, roll, accel_x, accel_y, accel_z
                yaw, pitch, roll, x, y, z = map(float, data.split(","))

                # Time step
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                # Apply acceleration threshold
                x = x if abs(x) > ACCEL_THRESHOLD else 0
                y = y if abs(y) > ACCEL_THRESHOLD else 0
                z = z if abs(z) > ACCEL_THRESHOLD else 0

                # Integrate acceleration to get velocity
                velocity_x = (velocity_x + x * dt) * DECAY_FACTOR 
                velocity_y = (velocity_y + y * dt) * DECAY_FACTOR
                velocity_z = (velocity_z + z * dt) * DECAY_FACTOR

                # Integrate velocity to get distance
                distance_x += velocity_x * dt
                distance_y += velocity_y * dt
                distance_z += velocity_z * dt

                # Print Yaw, Pitch, Roll & Distance
                print(f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")
                print(f"Distance -> X: {distance_x:.2f} m, Y: {distance_y:.2f} m, Z: {distance_z:.2f} m\n")
                time.sleep(0.5)
    except Exception as e:
        print("Error:", e)
```        

Keep in mind even though this method should work on paper it doesn't in real world scenarios, I am just adding this code just in case someone might need it to do a few tests.

