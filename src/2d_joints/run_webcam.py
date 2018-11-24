

import cv2

from estimator import TfPoseEstimator

e = TfPoseEstimator('models/graph/mobilenet_thin_432x368/graph_opt.pb', target_size=(432, 368))

if __name__ == '__main__':

    cam = cv2.VideoCapture(0)

    while True:
        ret_val, image = cam.read()
        humans = e.inference(image)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.imshow('tf-pose-estimation result', image)
        if cv2.waitKey(1) == 27:
            break
        # logger.debug('finished+')

    cv2.destroyAllWindows()
