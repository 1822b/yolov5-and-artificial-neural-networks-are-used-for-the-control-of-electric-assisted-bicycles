import RPi.GPIO as GPIO
import time
import numpy as np
import tflite_runtime.interpreter as tflite
from MPU6050 import MPU6050
import math

# 车轮参数
RADIUS = 0.325
CIRCUM = RADIUS * 2 * 3.14

# 引脚定义
SENSOR_PIN_SPEED = 11  # 速度传感器引脚
SENSOR_PIN_BPM = 12    # 踏频传感器引脚
PIN_PWM = 13           # 电机PWM引脚

# 初始化全局变量
last_triggered_time_speed = None
last_triggered_time_bpm = None
speed = 0  # 速度
bpm = 0    # 踏频

# 初始化GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SENSOR_PIN_SPEED, GPIO.IN)
GPIO.setup(SENSOR_PIN_BPM, GPIO.IN)
GPIO.setup(PIN_PWM, GPIO.OUT)
p_pwm = GPIO.PWM(PIN_PWM, 100)
p_pwm.start(0)

# MPU6050初始化
mpu = MPU6050(1, 0x68, -3954, 2485, 1264, -31, 33, -8, True)
mpu.dmp_initialize()
mpu.set_DMP_enabled(True)
packet_size = mpu.DMP_get_FIFO_packet_size()

# 加载TensorFlow Lite模型
interpreter = tflite.Interpreter(model_path="trained_model_best.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def sensor_callback_speed(channel):
    """速度传感器回调函数"""
    global last_triggered_time_speed, speed
    current_time = time.time()
    if last_triggered_time_speed:
        time_interval = current_time - last_triggered_time_speed
        frequency = 1 / time_interval
        speed = CIRCUM * frequency * 3.6  # 转换为km/h
    last_triggered_time_speed = current_time

def sensor_callback_bpm(channel):
    """踏频传感器回调函数"""
    global last_triggered_time_bpm, bpm
    current_time = time.time()
    if last_triggered_time_bpm:
        time_interval = current_time - last_triggered_time_bpm
        bpm = 60 / time_interval  # 转换为每分钟转速
    last_triggered_time_bpm = current_time

def read_mpu6050():
    """读取MPU6050数据"""
    FIFO_count = mpu.get_FIFO_count()
    mpu_int_status = mpu.get_int_status()
    if (FIFO_count == 1024) or (mpu_int_status & 0x10):
        mpu.reset_FIFO()
    elif (mpu_int_status & 0x02):
        while FIFO_count < packet_size:
            FIFO_count = mpu.get_FIFO_count()
        FIFO_buffer = mpu.get_FIFO_bytes(packet_size)
        roll_pitch_yaw = mpu.DMP_get_euler_roll_pitch_yaw(
            mpu.DMP_get_quaternion_int16(FIFO_buffer), 
            mpu.DMP_get_gravity(mpu.DMP_get_quaternion_int16(FIFO_buffer))
        )
        return roll_pitch_yaw.z  # 返回偏航角
    return None

def angle_to_percentage(angle_deg):
    """将角度转换为坡度百分比"""
    angle_rad = math.radians(angle_deg)
    return math.tan(angle_rad) * 100

class MotorController:
    def __init__(self, pwm_instance, interpreter):
        self.pwm = pwm_instance
        self.interpreter = interpreter
        self.output_ratio = 1.0

    def set_output_ratio(self, ratio):
        self.output_ratio = ratio

    def control_motor(self, speed, bpm, yaw):
        # 准备输入数据
        adjusted_input = np.array([[speed * self.output_ratio, bpm * self.output_ratio, yaw]], dtype=np.float32)
        self.interpreter.set_tensor(input_details[0]['index'], adjusted_input)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        # 限制PWM占空比
        pwm_value = min(output_data[0][0], 70)
        self.pwm.ChangeDutyCycle(pwm_value)
        print(f"Speed: {speed:.2f} km/h, BPM: {bpm:.2f}, Yaw: {yaw:.2f}%, PWM: {pwm_value:.2f}")

GPIO.add_event_detect(SENSOR_PIN_SPEED, GPIO.RISING, callback=sensor_callback_speed)
GPIO.add_event_detect(SENSOR_PIN_BPM, GPIO.RISING, callback=sensor_callback_bpm)

def main():
    motor_controller = MotorController(p_pwm, interpreter)
    try:
        while True:
            yaw_value = read_mpu6050()
            if yaw_value is None:
                yaw_value = 0
            yaw = angle_to_percentage(-yaw_value) - 20

            # 调用电机控制逻辑
            motor_controller.control_motor(speed, bpm, yaw)
            time.sleep(0.1)
    except KeyboardInterrupt:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
