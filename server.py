import os
import os.path as osp
import sys
BUILD_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "build/service/")
sys.path.insert(0, BUILD_DIR)
import argparse

import grpc
from concurrent import futures
import fib_pb2
import fib_pb2_grpc

import cv2
import argparse
import multiprocessing as mp
import mediapipe as mp2
mp_drawing = mp2.solutions.drawing_utils
mp_drawing_styles = mp2.solutions.drawing_styles
mp_pose = mp2.solutions.pose
mp_object_detection = mp2.solutions.object_detection
mp_hands = mp2.solutions.hands

flag  = "object"
draw_flag = "object"


class FibCalculatorServicer(fib_pb2_grpc.FibCalculatorServicer):

    def __init__(self):
        pass

    def Compute(self, request, context):
        global flag
        n = request.order
        state.value = n
        if n == 1: 
            flag = "object"
        elif n == 2: 
            flag = "hand"
        elif n == 3: 
            flag = "pose"
        print("now processing in %s mode" % flag)
        

        response = fib_pb2.FibResponse()
        response.value = 0

        return response
    
def object_detection(image):
    with mp_object_detection.ObjectDetection(min_detection_confidence=0.5) as object_detection:
        results = object_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
    return (image, results.detections)
    
def draw_pose (image):
    with mp_pose.Pose( min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    

    return image, results.pose_landmarks

def draw_hands (image):
    with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
    return image, results.multi_hand_landmarks

def gstreamer_camera(queue):
    # Use the provided pipeline to construct the video capture in opencv
    pipeline = (
        "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)960, height=(int)540, "
            "format=(string)NV12, framerate=(fraction)30/1 ! "
        "queue ! "
        "nvvidconv flip-method=2 ! "
            "video/x-raw, "
            "width=(int)960, height=(int)540, "
            "format=(string)BGRx, framerate=(fraction)30/1 ! "
        "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
        "appsink"
    )
    # Complete the function body
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    print(cap.isOpened())
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #print(time.strftime('%X'), frame.shape)
            queue.put(frame)
    except KeyboardInterrupt as e:
        cap.release()

def gstreamer_rtmpstream(queue):
    # Use the provided pipeline to construct the video writer in opencv
    pipeline = (
        "appsrc ! "
            "video/x-raw, format=(string)BGR ! "
        "queue ! "
        "videoconvert ! "
            "video/x-raw, format=RGBA ! "
        "nvvidconv ! "
        "nvv4l2h264enc bitrate=8000000 ! "
        "h264parse ! "  
        "flvmux ! "
        'rtmpsink location="rtmp://localhost/rtmp/live live=1"'
    )
    # Complete the function body
    # You can apply some simple computer vision algorithm here
    out = cv2.VideoWriter(pipeline, fourcc = 0, apiPreference = cv2.CAP_GSTREAMER, fps=30, frameSize = (960, 540))

    if not out.isOpened():
        print("hehe")
        exit()
    try:    
        count = 0
        count_max = 10 # detection will be performed per "count_max" frames.
        while True:
            if queue.empty():continue
            frame = queue.get()
            if count == 0:
                # print(state.value)
                if int(state.value) == 1:
                    frame, detections = object_detection(frame)
                    draw_flag = "object"
                elif  int(state.value) == 3:
                    frame, landmarks = draw_pose(frame)
                    draw_flag = "pose"
                elif  int(state.value) == 2:
                    frame, multi_hand_landmarks = draw_hands(frame)
                    draw_flag = "hand"
                count += 1
            else:
                count += 1
                if draw_flag == "object":
                    if detections:
                        for detection in detections:
                            mp_drawing.draw_detection(frame, detection)
                elif draw_flag == "pose":
                    mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                elif draw_flag == "hand":
                    if multi_hand_landmarks:
                        for hand_landmarks in multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                if count == count_max: 
                    count = 0
            out.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break
        out.release()
        cv2.destroyAllWindows()
    except: 
        out.release()
        cv2.destroyAllWindows()    

if __name__ == "__main__":
    q = mp.Queue(maxsize=3)
    state = mp.Value('d', 1)
    p = mp.Process(target=gstreamer_camera, args=[q])
    p2 = mp.Process(target=gstreamer_rtmpstream, args=[q])
    p.start()
    p2.start()

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8080, type=int)
    args = vars(parser.parse_args())

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FibCalculatorServicer()
    fib_pb2_grpc.add_FibCalculatorServicer_to_server(servicer, server)

    try:
        server.add_insecure_port(f"{args['ip']}:{args['port']}")
        server.start()
        print(f"Run gRPC Server at {args['ip']}:{args['port']}")
        server.wait_for_termination()
        p.terminate()
        p2.terminate()
        p.join()
        p2.join()
    except KeyboardInterrupt:
        pass
