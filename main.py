from detector.faceboxes import FaceboxesTensorflow
import configparser
import requests
import base64
import glob
import cv2
import os
from requests import Timeout
import json


def encode_image_base64(image, img_num):
    try:
        _, buffer = cv2.imencode('.png', image)
    except cv2.error as e:
        print("PNG encoding error on image{}.jpg {{\n".format(img_num) + "    image shape = {}\n}}".format(image))
        return None
    try:
        b64_image = base64.b64encode(buffer)
        b64_image_with_prefix = "data:image/png;base64," + b64_image.decode("utf-8")
        return b64_image_with_prefix
    except Exception:
        print("Base64 encoding error on image{}.jpg: ".format(img_num))
        return None


def send_packet(packet):
    headers = {'content-type': 'application/json; charset=UTF-8'}
    try:
        response = requests.post('http://localhost/api', data=json.dumps(packet),
                                 headers=headers, verify=False)
    except Timeout:
        print("Request Timed Out.")
    except ConnectionError:
        print("Connection error.")
    else:
        if response.status_code == 404:
            print("ATENTO error 404, not found")
            exit(1)
        else:
            try:
                print("ATENTO message: {}".format(response.json()["message"].encode('utf8')))
            except json.JSONDecodeError:
                print("json Decode Error")


def crop_detection(frame, video_W, video_H, left_x, top_y, right_x, bottom_y):
    aux_top_y = top_y
    aux_bottom_y = bottom_y
    aux_left_x = left_x
    aux_right_x = right_x

    if top_y - 5 < 0:
        aux_top_y = 0
    if bottom_y - 5 > video_H:
        aux_bottom_y = int(video_H)
    if left_x - 5 < 0:
        aux_left_x = 0
    if right_x + 5 > video_W:
        aux_right_x = int(video_W)

    crop = frame[aux_top_y:aux_bottom_y, aux_left_x:aux_right_x]
    return crop


def send_to_api(image, crop, og_img_num, device_name):
    packet = {
        "requestNumber": 00,
        "companyCode": 4,
        "dispositiveType": 2,
        "captureDeviceCode": device_name,
        "appCode": 7,
        "latitude": "null",
        "longitude": "null",
        "truePictureTree": "99",
        "eventName": "image{}.jpg".format(og_img_num),
        "flagFace": 1,
        "personalType": 1,
        "trueImage": encode_image_base64(image, og_img_num),
        "cropFace": encode_image_base64(crop, og_img_num)
    }
    send_packet(packet)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    path = config['Default']['path_directory']
    if not path.endswith('/'):
        path = path + '/'
    MOVE = config['Default'].getboolean('move')
    OPTION = config['Default'].getint('option')
    device_name = config['Default']['device_name']

    try:
        arquivo = open("total_faces.txt", 'r')
        num_faces = int(arquivo.readline())
    except OSError:
        num_faces = 0
    except ValueError:
        num_faces = 0
    if OPTION == 1 or OPTION == 3:
        # Instantiate detector
        detector = FaceboxesTensorflow(model_path=config['Detector']['weights'],
                                       score_threshold=config['Detector'].getfloat('score_threshold'))
        all_image_paths = []
        for r, d, f in os.walk(path):
            for file in f:
                if file.endswith(".jpg"):
                    all_image_paths.append(os.path.join(r, file))

        total_imgs = len(all_image_paths)
        for i in range(total_imgs):

            # Get image original path
            image_og_path = all_image_paths.pop(0)
            # Move and rename this file
            new_image_num = str(i).zfill(6)
            new_image_path = 'images_out/image{}.jpg'.format(new_image_num)
            os.rename(image_og_path, new_image_path)
            # Detect faces
            image = cv2.imread(new_image_path)
            boxes, _ = detector.detect(image)
            img_h, img_w, _ = image.shape
            num_faces += len(boxes)
            for num, box in enumerate(boxes):
                crop = crop_detection(image, img_w, img_h, box[0], box[1], box[2], box[3])
                crop_name = 'face{}'.format(str(num).zfill(3))

                # [3] Extrair faces e importar para o atento
                if OPTION == 3:
                    send_to_api(image, crop, new_image_num, device_name=device_name)

                # [1] Extrair faces e armazenar em disco
                else:
                    cv2.imwrite('faces_out/image{}-{}.jpg'.format(new_image_num, crop_name), crop)

            print("Processed images: {}/{}".format(i, total_imgs))
            print("Faces: {}".format(num_faces))
            if num_faces > 400000:
                break
        arquivo = open("total_faces.txt", "w")
        arquivo.write(str(num_faces))

    elif OPTION == 2:
        all_image_paths = []
        for filename in glob.glob(os.path.join("faces_out/", '*.jpg')):
            all_image_paths.append(filename)

        for i in range(len(all_image_paths)):
            crop_path = all_image_paths.pop(0)
            crop_name = crop_path.split('/')[-1]
            from_image = crop_name.split('-')[0]
            # Attempt to load original image
            original_image = cv2.imread("images_out/{}.jpg".format(from_image))
            if original_image is None:
                print("original image {} not found".format(from_image))
                continue
            crop = cv2.imread(crop_path)
            if crop is None:
                print("face image {} not found".format(crop_name))
            # If we found both images, send them to atento
            send_to_api(original_image, crop, from_image[5:], device_name=device_name)
    else:
        print("Options must be 1, 2 or 3.")
        exit(0)
