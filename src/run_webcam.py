import cv2
import common
import torch
import numpy as np
import sys
import json
import threading

from websocket_server import WebsocketServer

from model2 import Model
from enum import Enum
from estimator import TfPoseEstimator
from utils import update_lr, unnormalize, sort_ckpt, normalize
from torch.autograd import Variable
sys.path.append("../")

e = TfPoseEstimator('models/graph/mobilenet_thin_432x368/graph_opt.pb', target_size=(432, 368))

netOutMap = [
    'head',
    'centerShoulder',
    'rightShoulder',
    'rightElbow',
    'rightWrist',
    'leftShoulder',
    'leftElbow',
    'leftWrist',
    'rightHip',
    'rightKnee',
    'rightAnkle',
    'leftHip',
    'leftKnee',
    'leftAnkle',
]

indexMap2d = [
    'centerHip',
    'leftHip',
    'leftKnee',
    'leftAnkle',
    'rightHip',
    'rightKnee',
    'rightAnkle',
    'spine',
    'centerShoulder',
    'head',
    'leftShoulder',
    'leftElbow',
    'leftWrist',
    'rightShoulder',
    'rightElbow',
    'rightWrist'
];

cuda = torch.cuda.is_available()
model = Model().cuda() if cuda else Model()
stat_2d = torch.load('../data/stat_2d.pth.tar')
stat_3d = torch.load('../data/stat_3d.pth.tar')

ckpt = torch.load('../checkpoint/best.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(ckpt['model'])

cam = cv2.VideoCapture(0)
ret_val, image = cam.read()
image_h, image_w = image.shape[:2]

def new_client(client, server):
    print("client connected")
    server.send_message_to_all("Hey all, a new client has joined us")

# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])

def start_server():
    server.run_forever()

server = WebsocketServer(8001, host='localhost')
server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
print("Server start, print Ctrl + C to quit")

server_thread = threading.Thread(target=start_server).start()

def pose_detection(input):
    model.eval()
    data = np.array(input).reshape((1, -1))

    used_data = normalize(data, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
    used_data = np.repeat(used_data, 64, axis=0)
    used_data = torch.from_numpy(used_data).float()
    used_data = Variable(used_data.cuda()) if cuda else Variable(used_data.cpu())

    output = model(used_data)

    unnormal_output = unnormalize(output.data.cpu(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
    used_output = unnormal_output[:, stat_3d['dim_use']].reshape((-1, 3)).reshape((-1, 16, 3))
    output_3d = used_output[0].reshape((1, -1))

    return output_3d

def joints_data(human):
    joints = [[human.body_parts[each].x * image_w, human.body_parts[each].y * image_h, netOutMap[each]] for
              each in human.body_parts if each in range(len(netOutMap))]
    if len(joints) != len(netOutMap):
        return []

    input_joints = [None for i in range(len(indexMap2d))]

    for index, each in enumerate(joints):
        input_joints[indexMap2d.index(each[2])] = each[:-1]

    left_hip = joints[netOutMap.index('leftHip')]
    right_hip = joints[netOutMap.index('rightHip')]

    input_joints[indexMap2d.index('centerHip')] = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]

    center_hip = input_joints[indexMap2d.index('centerHip')]
    center_shoulder = input_joints[indexMap2d.index('centerHip')]

    input_joints[indexMap2d.index('spine')] = [(center_hip[0] + center_shoulder[0]) / 2,
                                               (center_hip[1] + center_shoulder[1]) / 2]

    return input_joints

if __name__ == '__main__':
    while True:
        ret_val, image = cam.read()
        humans = e.inference(image)

        if len(humans) > 0:
            joints = joints_data(humans[0])
            if len(joints) == 16:
                input_vector = []
                for each in joints:
                    input_vector.append(each[0])
                    input_vector.append(each[1])
                server.send_message_to_all(json.dumps(pose_detection(input_vector).tolist()[0]))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.imshow('tf-pose-estimation result', image)
        if cv2.waitKey(1) == 27:
            break
        # logger.debug('finished+')

    cv2.destroyAllWindows()
