from detector.faceboxes import FaceboxesTensorflow
from requests import Timeout
import configparser
import requests
import base64
import json
import glob
import cv2
import os


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


def call_funcao_07(image, crop, og_img_num, device_name):
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


def call_funcao_02(image, image_name, device_name):
    packet = {
        "companyCode": 4,
        "captureDeviceCode": device_name,
        "appCode": 7,
        "requestNumber": 00,
        "truePictureTree": 99,
        "keyPerson": image_name,
        "validatePictureTree": 99,
        "imageValidate": encode_image_base64(image, image_name),
        "latitude": "null",
        "longitude": "null"
    }
    pass
    send_packet(packet)


def parse_config():
    config = configparser.ConfigParser()
    config.read("config.ini")

    path = config['Default']['path_directory']
    if not path.endswith('/'):
        path = path + '/'
    move = config['Default'].getboolean('move')
    option = config['Default'].getint('option')
    device = config['Default']['device_name']
    config_dict = {
        "path": path,
        "move": move,
        "option": option,
        "device": device,
        "weights_path": config['Detector']['weights'],
        "score_threshold": config['Detector'].getfloat('score_threshold')
    }

    try:
        arquivo = open("total_faces.txt", 'r')
        num_faces = int(arquivo.readline())
    except OSError:
        num_faces = 0
    except ValueError:
        num_faces = 0

    config_dict["num_faces"] = num_faces

    return config_dict


def get_all_image_paths(path):
    all_image_paths = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(".jpg"):
                all_image_paths.append(os.path.join(r, file))
    return all_image_paths


if __name__ == '__main__':
    config = parse_config()

    if config["option"] == 1 or config["option"] == 3:
        # Instantiate detector
        detector = FaceboxesTensorflow(model_path=config["weights_path"],
                                       score_threshold=config["score_threshold"])
        all_image_paths = get_all_image_paths(config["path"])
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
            config["num_faces"] += len(boxes)
            for num, box in enumerate(boxes):
                crop = crop_detection(image, img_w, img_h, box[0], box[1], box[2], box[3])
                crop_name = 'face{}'.format(str(num).zfill(3))

                # [3] Extrair faces e importar para o atento
                if config["option"] == 3:
                    call_funcao_07(image, crop, new_image_num, device_name=config["device"])

                # [1] Extrair faces e armazenar em disco
                else:
                    cv2.imwrite('faces_out/image{}-{}.jpg'.format(new_image_num, crop_name), crop)

            print("Processed images: {}/{}".format(i, total_imgs))
            print("Faces: {}".format(config["num_faces"]))
            if config["num_faces"] > 400000:
                break
        with open("total_faces.txt", "w") as arquivo:
            arquivo.write(str(config["num_faces"]))

    elif config["option"] == 2:
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
            call_funcao_07(original_image, crop, from_image[5:], device_name=config["device"])

    elif config["option"] == 4:
        all_image_paths = get_all_image_paths(config["path"])
        total_imgs = len(all_image_paths)

        for i in range(total_imgs):
            image_og_path = all_image_paths.pop(0)
            image_name = image_og_path.split('/')[-1]
            new_image_path = 'funcao_02/{}'.format(image_name)
            # read image
            image = cv2.imread(image_og_path)
            # call fun 02
            call_funcao_02(image, image_name, config["device"])
            # Once we're done with this image, move it
            os.rename(image_og_path, new_image_path)
            # Update status
            print("Image: {}. Total:{}/{}.".format(image_name, i, total_imgs))
    else:
        print("Options must be 1, 2, 3 or 4.")
        exit(0)
