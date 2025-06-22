import sys
import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import math
import RPi.GPIO as GPIO
from MPU6050 import MPU6050
import random
import tflite_runtime.interpreter as tflite
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
yaw = 0
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

with open("detect_test_0%.csv", "w") as file:
    file.write("Timestamp,Speed(km/h),BPM,PWM\n")

def sensor_callback_speed(channel):
    """速度传感器回调函数"""
    global last_triggered_time_speed, speed
    current_time = time.time()
    if last_triggered_time_speed:
        time_interval = current_time - last_triggered_time_speed
        frequency = 1 / time_interval
        speed = CIRCUM * frequency * 3.6  # 转换为km/h
    last_triggered_time_speed = current_time

# def sensor_callback_bpm(channel):
#     """踏频传感器回调函数"""
#     global last_triggered_time_bpm, bpm
#     current_time = time.time()
#     if last_triggered_time_bpm:
#         time_interval = current_time - last_triggered_time_bpm
#         bpm = 60 / time_interval  # 转换为每分钟转速
#     last_triggered_time_bpm = current_time

# def read_mpu6050():
#     """读取MPU6050数据"""
#     FIFO_count = mpu.get_FIFO_count()
#     mpu_int_status = mpu.get_int_status()
#     if (FIFO_count == 1024) or (mpu_int_status & 0x10):
#         mpu.reset_FIFO()
#     elif (mpu_int_status & 0x02):
#         while FIFO_count < packet_size:
#             FIFO_count = mpu.get_FIFO_count()
#         FIFO_buffer = mpu.get_FIFO_bytes(packet_size)
#         roll_pitch_yaw = mpu.DMP_get_euler_roll_pitch_yaw(
#             mpu.DMP_get_quaternion_int16(FIFO_buffer), 
#             mpu.DMP_get_gravity(mpu.DMP_get_quaternion_int16(FIFO_buffer))
#         )
#         return roll_pitch_yaw.z  # 返回偏航角
#     return None

class MotorController:
    def __init__(self, pwm_instance, interpreter, id):
        self.pwm = pwm_instance
        self.interpreter = interpreter
        self.id = id
        self.start_time = time.time()  # 记录程序启动时刻

    def control_motor(self, speed, bpm, yaw):
        # 准备输入数据
        elapsed_time = time.time() - self.start_time
        adjusted_input = np.array([[speed , bpm , yaw]], dtype=np.float32)
        self.interpreter.set_tensor(input_details[0]['index'], adjusted_input)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        pwm_value = min(output_data[0][0], 70)
        # 限制PWM占空
        if self.id == 0:
            pwm_value_1 = pwm_value * 0.3
        elif self.id == 1:
            pwm_value_1 = pwm_value * 0.5
        elif self.id is None:  # 未检测到目标
            pwm_value_1 = pwm_value  # 保持原始 speed（或进行其他处理）
        self.pwm.ChangeDutyCycle(pwm_value_1)
        print(f"Speed: {speed:.2f} km/h, BPM: {bpm:.2f}, Yaw: {yaw:.2f}%, PWM: {pwm_value_1:.2f}")
        with open("detect_test_0%.csv", "a") as file:
            file.write(f"{elapsed_time:.2f},{speed:.2f},{bpm:.2f},{pwm_value_1:.2f}\n")
        # time.sleep(0.2)

GPIO.add_event_detect(SENSOR_PIN_SPEED, GPIO.RISING, callback=sensor_callback_speed)
# GPIO.add_event_detect(SENSOR_PIN_BPM, GPIO.RISING, callback=sensor_callback_bpm)

# def angle_to_percentage(angle_deg):
#     """将角度转换为坡度百分比"""
#     angle_rad = math.radians(angle_deg)
#     return math.tan(angle_rad) * 100

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

### YOLOv5 目标检测相关函数 ###
def initialize_yolo_model(model_path):
    """加载 YOLO 模型"""
    so = ort.SessionOptions()
    return ort.InferenceSession(model_path, so)

def infer_yolo(img, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.5):
    """推理图片并获取目标检测结果"""
    img_resized = cv2.resize(img, (model_w, model_h), interpolation=cv2.INTER_AREA)
    img_normalized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0)

    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
    outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)
    img_h, img_w, _ = img.shape
    return post_process_opencv(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)

def cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride):
    """计算输出的具体位置和大小"""
    grid = [np.zeros(1)] * nl
    row_ind = 0
    for i in range(nl):
        h, w = int(model_w / stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)
        outs[row_ind:row_ind + length, 0:2] = (
            outs[row_ind:row_ind + length, 0:2] * 2.0 - 0.5 + np.tile(grid[i], (na, 1))
        ) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (
            (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(anchor_grid[i], h * w, axis=0)
        )
        row_ind += length
    return outs

def _make_grid(nx, ny):
    """生成网格"""
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    """后处理目标检测结果"""
    conf = outputs[:, 4]
    c_x = outputs[:, 0] / model_w * img_w
    c_y = outputs[:, 1] / model_h * img_h
    w = outputs[:, 2] / model_w * img_w
    h = outputs[:, 3] / model_h * img_h
    p_cls = outputs[:, 5:]
    cls_id = np.argmax(p_cls, axis=1)

    p_x1 = np.expand_dims(c_x - w / 2, -1)
    p_y1 = np.expand_dims(c_y - h / 2, -1)
    p_x2 = np.expand_dims(c_x + w / 2, -1)
    p_y2 = np.expand_dims(c_y + h / 2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

    ids = cv2.dnn.NMSBoxes(areas.tolist(), conf.tolist(), thred_cond, thred_nms)
    if len(ids) > 0:
        ids = ids.flatten()
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    return [], [], []

### 综合功能 ###
class VideoDetectionControl:
    def __init__(self, yolo_model, label_dict, motor_controller, interpreter):
        self.yolo_model = yolo_model
        self.label_dict = label_dict
        self.flag_det = False
        self.motor_controller = motor_controller  # 电机控制器
        self.interpreter = interpreter  # 神经网络模型解释器

        # 初始化标志位字典，设置不同物品对应的标志位
        self.item_flags = {
            0: 'people',  # 假设 ID 0 代表人物，设置标志位为 'people'
            1: 'car',     # 假设 ID 1 代表汽车，设置标志位为 'car'
            
        }

        self.detected_flags = {}  # 存储当前检测到的标志位

        # 视频捕获
        self.cap = cv2.VideoCapture(0)

    def update_frame(self):
        """更新视频帧并检测物品"""
        success, img0 = self.cap.read()
        if success:
            # if self.flag_det:
                # YOLO 检测
            boxes, scores, ids = infer_yolo(img0, *self.yolo_model)
            if len(ids) > 0:
                for box, score, cls_id in zip(boxes, scores, ids):
                    label = f"{self.label_dict[cls_id]}: {score:.2f}"
                    plot_one_box(box.astype(np.int16), img0, label=label)

                    # 根据物品 ID 设置标志位
                    if cls_id in self.item_flags:
                        item_flag = self.item_flags[cls_id]
                        self.detected_flags[item_flag] = True  # 设置为检测到的标志位
                        self.motor_controller.id = cls_id  # 将物体类别的 ID 设置到电机控制器
                        print(f"Detected ID: {self.motor_controller.id}")
            else:
                self.motor_controller.id = None  # 没有检测到目标时清空 id 的值
                print(f"Detected ID: {self.motor_controller.id}")
            # 读取 MPU6050 传感器数据并控制电机
            # yaw_value = read_mpu6050()
            bpm = random.uniform(50, 70)
            yaw = 0
            # if yaw_value is None:
            #     yaw_value = 0
            # yaw = angle_to_percentage(-yaw_value) - 20  # 转换为电机控制的百分比
            # 控制电机
            self.motor_controller.control_motor(speed, bpm, yaw)

            # 图像处理和显示
            cv2.imshow('Video Detection', img0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()

    def get_detected_flags(self):
        """获取当前检测到的标志位"""
        return self.detected_flags

### 主函数 ###
if __name__ == "__main__":
    # YOLO 模型参数
    yolo_model_path = "car-people-best.onnx"
    label_dict = {0: "people", 1: "car"}
    model_h, model_w, nl, na, stride = 320, 320, 3, 3, [8., 16., 32.]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
    yolo_model = initialize_yolo_model(yolo_model_path), model_h, model_w, nl, na, stride, anchor_grid
    motor_id = None
    motor_controller = MotorController(p_pwm, interpreter, motor_id)  # 初始化电机控制器
    # 创建 VideoDetectionControl 实例
    video_detection_control = VideoDetectionControl(yolo_model, label_dict, motor_controller, interpreter)
    
    while True:
        video_detection_control.update_frame()