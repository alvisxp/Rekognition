import boto3
import cv2
import os

import credentials

output_dir = './data'
output_dir_imgs = os.path.join(output_dir, 'imgs')
output_dir_anns = os.path.join(output_dir, 'anns')

rekognition_client = boto3.client('rekognition', aws_access_key_id=credentials.access_key,
                                  aws_secret_access_key=credentials.secret_key)

target_class = 'Zebra'

cap = cv2.VideoCapture('./zebras.mp4')

frame_number = -1

ret = True
while ret:

    ret, frame = cap.read()

    if ret:

        frame_number += 1
        H, W, _ = frame.shape

        _, buffer = cv2.imencode('.jpg', frame)

        image_bytes = buffer.tobytes()

        response = rekognition_client.detect_labels(Image={'Bytes': image_bytes}, MinConfidence=50)

        with open(os.path.join(output_dir_anns, 'frame_{}.txt'.format(str(frame_number).zfill(6))), 'w') as f:
            for label in response['Labels']:
                if label['Name'] == target_class:
                    for instance_number in range(len(label['Instances'])):
                        box = label['Instances'][instance_number]['BoundingBox']
                        x1 = box['Left'] * W
                        y1 = box['Top'] * H
                        width = box['Width'] * W
                        height = box['Height'] * H
                        print(x1, y1, width, height)

                        f.write('{} {} {} {} {}\n'.format(0,
                                                          (x1 + width/2),
                                                          (y1 + height/2),
                                                          width,
                                                          height)
                                )

            f.close()

        cv2.imwrite(os.path.join(output_dir_imgs, 'frame_{}.jpg'.format(str(frame_number).zfill(6))), frame)

