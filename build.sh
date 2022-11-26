 python inference_onnx.py --model-path yolov7-p6-bonefracture.onnx  --img-path ./input/An-x-ray-showing-the-metacarpal-bone-fracture-at-right-fifth-bone-of-the-patient-on-the.png --dst-path ./input

docker build --no-cache -t anilyerramasu/yolo_wrist_fracture .

docker run --gpus all --ipc=host --rm -p 8000:5000 -v $(pwd)/input:/opt/output anilyerramasu/yolo_wrist_fracture 
