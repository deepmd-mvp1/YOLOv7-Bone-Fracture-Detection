# Methods for prediction for this competition
import os
import sys
import cv2
import numpy as np
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import subprocess
import json
import shutil
from flask import jsonify, send_file
import tempfile
from flask_cors import CORS
from inference_onnx import model_inference, post_process
import onnxruntime
from matplotlib.colors import TABLEAU_COLORS




app=Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = "/home/input"
# os.path.join(path, 'uploads')

# Make directory if uploads is not exists
# if not os.path.isdir(UPLOAD_FOLDER):
#     os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
device="cuda"
model_path="yolov7-p6-bonefracture.onnx"
providers = ['CUDAExecutionProvider'] if device=="cuda" else ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(model_path, providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name



def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]

colors = color_list()

def xyxy2xywh(bbox, H, W):

    x1, y1, x2, y2 = bbox

    return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

def load_img(img_file, img_mean=0, img_scale=1/255):
    img = cv2.imread(img_file)[:, :, ::-1]
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img


def post_process(img_file, output, score_threshold=0.3, format="xywh"):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    assert format == "xyxy" or format == "xywh"

    det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
    id2names = {
        0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
        3: "fracture", 4: "metal", 5: "periostealreaction", 
        6: "pronatorsign", 7:"softtissue", 8:"text"
    }

    img = cv2.imread(img_file)
    H, W = img.shape[:2]
    h, w = 640, 640
    label_txt = ""

    for idx in range(len(det_bboxes)):
        if det_scores[idx]>score_threshold:
            bbox = det_bboxes[idx]
            bbox = bbox @ np.array([[W/w, 0, 0, 0], [0, H/h, 0, 0], [0, 0, W/w, 0], [0, 0, 0, H/h]])
            bbox_int = [int(x) for x in bbox]
            label = det_labels[idx]
            
            if format=="xywh":
                bbox = xyxy2xywh(bbox, H, W)
            label_txt += f"{int(label)} {det_scores[idx]:.5f} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f}\n"

            color_map = colors[int(label)]
            txt = f"{id2names[label]} {det_scores[idx]:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), color_map, 2)
            cv2.rectangle(img, (bbox_int[0]-2, bbox_int[1]-text_height-10), (bbox_int[0] + text_width+2, bbox_int[1]), color_map, -1)
            cv2.putText(img, txt, (bbox_int[0], bbox_int[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img, label_txt


@app.route('/wrist', methods=['GET'])
def upload_form():
    return render_template('upload.html')

@app.route('/wrist/predict', methods=['POST'])
def Prediction():
   
    # os.mkdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':

        files = request.files.getlist('files[]')
        inputDir = tempfile.mkdtemp(dir="./output")
        print("input file + " + inputDir)
        for file in files:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(inputDir +"/" +filename)
            img = load_img(inputDir +"/" +filename)
            out = session.run([output_name], {input_name: img})

            output = output[0][:, :6]
            out_img, out_txt = post_process(inputDir +"/" +filename, out, 0.3, "xywh")
            cv2.imwrite((inputDir +"/" + "pred.jpg"), out_img[..., ::-1])

            return send_file(inputDir +"/" + "pred.jpg", mimetype="image/jpg")
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=6000,debug=False,threaded=True)