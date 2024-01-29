import boto3
import cv2
import os

import credentials

client = boto3.client('rekognition', aws_access_key_id=credentials.access_key,
                      aws_secret_access_key=credentials.secret_key)

cap = cv2.VideoCapture("./car.mp4")

ret = True
while ret:
    ret, frame = cap.read()
    H, W, _ = frame.shape

    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    response = client.detect_labels(Image={'Bytes': image_bytes}, MinConfidence=0.90)

    for label in response['Labels']:
        for instance_number in range(len(label['Instances'])):
            bbox = label['Instances'][instance_number]['BoundingBox']
            x1 = int(bbox['Left'] * W)
            y1 = int(bbox['Top'] * H)
            width = int(bbox['Width'] * W)
            height = int(bbox['Height'] * H)

            label_name = label['Name']
            confidence = label['Instances'][instance_number]['Confidence']

            cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (255, 0, 0))
            label_text = f"{label_name}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)