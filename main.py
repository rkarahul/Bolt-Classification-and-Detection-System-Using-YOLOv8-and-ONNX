####################################working in production calssifier cnnn###################################################
from datetime import datetime
import onnxruntime as ort
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
from PIL import Image
import time
import os
import uvicorn
from time import strftime
from torchvision import transforms
import torch
from ultralytics import YOLO
from collections import Counter
import traceback
import copy
# # Loading yolov8 classifer
# model = YOLO('best.pt')  # Load YOLOv8 model
# tempimg=cv2.imread("temp.png")
# res=model.predict(tempimg)
# print("temp done : ",res) #self.model.to('cpu')  # Ensure model runs on CPU


# Path to the ONNX model
onnx_model_path ='bolt_classifier_2025_old.onnx' #'tvs_norm_classifier_new.onnx'#
# Load the ONNX model using onnxruntime and specify the CPU execution provider
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
# Define the image transform (match the input preprocessing used for training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),        # Resize image to 224x224
    transforms.ToTensor(),                # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
])

def save_img(img):
    # Get the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Define the directory name with the current date and time
    img_name = f"image_{current_time}.png"
    folder_name=r"D:\bolt_vip_ cnn_cl\NG_Image"
    
    # Define the full path for the image
    image_path = os.path.join(folder_name, img_name)
    
    # Save the image
    cv2.imwrite(image_path, img)

class DetectionTask:
    def __init__(self, model_path, class_file):
        self.net = self.load_model(model_path)
        self.classes = self.load_classes(class_file)

    def load_model(self, model_path):
        return cv2.dnn.readNetFromONNX(model_path)

    def load_classes(self, class_file):
        with open(class_file, 'r') as file:
            return [line.strip() for line in file.readlines()]
    

class TargetDetectionTask:

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = YOLO('best.pt')  # Load YOLOv8 model
        self.model.to('cpu')  # Ensure model runs on CPU
        # Load the YOLOv8 model seg
        self.modelseg = YOLO('Segbest.pt') #('tvs_norm.pt')#

    def ImageEnhancer(self,image,cls):

        # Convert the result image to a PIL Image for further enhancement
        result_pil_image = Image.fromarray(image)

        # Adjust the brightness
        if cls=="NORM":
            #print("bright")
            factor = 0.8#1.3  # fator greater then 1 increase brightness , lees then 1 decrease the brithness
        else:
            #print("dark")
            factor = 0.7  # fator greater then 1 increase brightness , lees then 1 decrease the brithness
        enhancer = ImageEnhance.Brightness(result_pil_image)
        img_enhanced = enhancer.enhance(factor)
        # Convert back to NumPy array if needed for further processing
        proframe = np.array(img_enhanced)
        return proframe

    def find_idx_lower(self,res_list):
        unique_values = set(res_list)
        value_count={value:res_list.count(value) for value in unique_values}
        lower_count = min(value_count, key=value_count.get)
        idx_lower=[idx for idx, value in enumerate(res_list) if value == lower_count]
        return idx_lower

    def nut_classifier_seg(self,crop_image):
        class_label="not_nut"
        #crop_image = frame[y1:y1 + h, x1:x1 + w]
        if crop_image is None:
            print(f"Failed to read cropimage none")
        area_mask=[]
        results = self.modelseg.predict(crop_image,conf=0.25,iou=0.1,imgsz=640, verbose=False,device="cpu")#save=True, imgsz=320, conf=0.5,iou=.7,device=device
        # Extract bounding boxes, classes, names, and confidences
        #boxes = results[0].boxes.xyxy.tolist()#[x_min, y_min, x_max, y_max]
        classes = results[0].boxes.cls.tolist()
        names = results[0].names #{0: 'NORM', 1: 'TVS'}
        if len(classes)==1:
            class_label=names[int(classes[0])]
            print("class_label : ",class_label)
            return class_label 
        #masks = results[0].masks
        confidences = results[0].boxes.conf.tolist()
        p=0
        for cls, conf in zip(classes, confidences): #conf
            #x_min, y_min, x_max, y_max = map(int, box)
            class_label = names[int(cls)]
            # try:
            #     polygon = masks[p].xy[0]
            #     polyarea=calculate_polygon_area(polygon)
            #     print("polyarea",polyarea)
            #     area_mask.append(polyarea)
            #     p+=1
            #     # Find the minimum area rectangle
            #     rect = cv2.minAreaRect(polygon) #(center(x, y), (width, height), angle of rotation)
            #     #print("min area rect",rect)       
            # except Exception as e:
            #     #poly_area.append(polyarea)
            #     print(traceback.format_exc())
            #     print("no detection")
            #     return class_label
        print("class_label : ",class_label)
        return class_label

    def perform_classifier_yolo(self, frame):

        # Perform inference (no need for PIL conversion)
        results = self.model.predict(frame)  
        #print("yolo results : ",results)
        probs = getattr(results[0], 'probs', None)
        if probs is not None:
            probs = probs.data.cpu()  # Ensure tensor is on CPU
            #print("probs : ",probs)
            max_idx = torch.argmax(probs).item()  # Get highest probability index
            max_confidence = probs[max_idx].item()

            # Ensure 'names' is correctly indexed
            class_name = results[0].names.get(max_idx, "Unknown") if hasattr(results[0], 'names') else "Unknown"
            class_name = class_name.upper()
            return [(class_name, max_confidence)]
        return [("Ok", 0.0)]

    def perform_classifier(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
        input_image = transform(image).unsqueeze(0).numpy()

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        outputs = session.run([output_name], {input_name: input_image})
        print("outputs : ",outputs)
        prediction = outputs[0]
        print("prediction : ",prediction)

        threshold = 3.8
        if prediction[0] > threshold:
            return "NORM"
        else:
            return "TVS"
        
    def performaug_old(self, img, cls):
        temp_res=[cls]
        res=self.perform_classifier_yolo(img)
        print("res : ",res)
        temp_res.append(res)
        #cv2.imwrite(r"D:\bolt_vip_ cnn_cl\withoutinh.png",img)
        inhanced_img=self.ImageEnhancer(img,cls)
        #cv2.imwrite(r"D:\bolt_vip_ cnn_cl\innhanced.png",inhanced_img)
        res_inh=self.perform_classifier(inhanced_img)
        print("res_inh : ",res_inh)
        temp_res.append(res_inh)
        counter=Counter(temp_res)
        final_res, _ =counter.most_common(1)[0]
        print(final_res)
        return final_res
    
    def performaug(self, img, cls):
        temp_res = [cls]  # cls is likely a string

        res = self.perform_classifier_yolo(img)  # Returns a list of tuples
        if res:
            temp_res.append(res[0][0])  # Extract only the class name (string)

        #inhanced_img = self.ImageEnhancer(img, cls)
        #res_inh = self.perform_classifier(inhanced_img)  # This is already a string
        res_inh=self.nut_classifier_seg(img)
        temp_res.append(res_inh)
        print("temp res : ",temp_res)
        counter = Counter(temp_res)  # ✅ No more TypeError
        final_res, _ = counter.most_common(1)[0]

        return temp_res[1]#final_res

        

class CropDetectionTask(DetectionTask):
    def perform_detection(self, frame, confidence_threshold=0.65, nms_threshold=0.1):
        if len(frame.shape) == 2:
            height, width = frame.shape
            channels = 1
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()[0]

        class_ids = []
        confidences = []
        boxes = []
        x_scale, y_scale = width / 640, height / 640

        for row in detections:
            confidence = row[4]
            if confidence > confidence_threshold:
                classes_score = row[5:]
                class_id = np.argmax(classes_score)
                if classes_score[class_id] > confidence_threshold:
                    class_ids.append(class_id)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx - w/2) * x_scale)
                    y1 = int((cy - h/2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1, y1, width, height])
                    if width >= 150 and height >= 150:
                        boxes.append(box)
        # Ensure there are valid detections before applying NMS
        if not boxes or not confidences:
            print("No valid detections found.")
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        return [(boxes[i], self.classes[class_ids[i]], confidences[i]) for i in indices]


    def draw_boxes(self, frame, crop_detections, target_task):
        detected_labels = []

        results = {}  # Store processed results for efficiency

        for box, label, confidence in crop_detections:
            x1, y1, w, h = box
            roi = frame[y1:y1+h, x1:x1+w]

            target_level = target_task.perform_classifier(roi)  # Single call
            print("target_detection:", target_level)
            #results.append((box, target_level))
            results[(x1,y1,w,h)] = target_level
            detected_labels.append(target_level)

        try:

            if len(set(detected_labels)) > 1:
                idx_lower=target_task.find_idx_lower(detected_labels)
                items_res=list(results.items())
                for idx in idx_lower:
                    boxx,labN=items_res[idx]
                    x11, y11, w1, h1=boxx
                    imgA=frame[y11:y11+h1, x11:x11+w1]
                    res_aug=target_task.performaug(imgA,labN)
                    detected_labels[idx]=res_aug
        except Exception as e:
            print(traceback.format_exc())

        # Determine final color based on detected labels
        final_color = (0, 255, 0) if len(detected_labels) == 16 and len(set(detected_labels)) == 1 else (0, 0, 255)
        i=0
        # Draw results in a single loop
        for box, label in results.items():
            x1, y1, w, h = box
            label=detected_labels[i]
            i+=1
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), final_color, 4)
            frame = cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, final_color, 2, cv2.LINE_AA)

        return frame, detected_labels


def process_image_with_classificationD1(img):

    if img is None:
        raise ValueError("Input image is None")

    class_result = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    cnts, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []
    results={}
    for i, (c, hier) in enumerate(zip(cnts, hierarchy)):
        x, y, w, h_ = cv2.boundingRect(c)

        if (w <= 380 and h_ <= 380) and (w >= 220 and h_ >= 220):
            ROI = img[y:y + h_, x:x + w]
            dst_avg = np.mean(ROI)
            label = "NORM" if dst_avg > 100 else "TVS"
            class_result.append(label)
            results[(x, y,w,h_)] = label

    # Draw rectangle
    final_color = (0, 255, 0) if len(class_result) == 16 and len(set(class_result)) == 1 else (0, 0, 255)
    i=0
    # Draw results in a single loop
    for box, label in results.items():
        x1, y1, w, h = box
        label=class_result[i]
        i+=1
        img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), final_color, 4)
        img = cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, final_color, 2, cv2.LINE_AA)

    return img, class_result

app = FastAPI()

class ImageData(BaseModel):
    image: str

@app.get("/ServerCheck")
def read_root():
    return {"status": "server is running"}

def score_img(frame, crop_task, target_task):
    crop_detections = crop_task.perform_detection(frame)
    frame_with_detections, class_result = crop_task.draw_boxes(frame, crop_detections, target_task)

    if not isinstance(frame_with_detections, np.ndarray):
        raise ValueError("The processed image is not a valid NumPy array")

    return frame_with_detections, class_result

# load the model object
crop_task = CropDetectionTask('gear_bolt.onnx', 'crop.names')
target_task = TargetDetectionTask()

#temp detections
temp_img=cv2.imread("temp_check.bmp")
p_timg, result_tempImg = score_img(temp_img, crop_task, target_task)
print("Temp detction done : ", result_tempImg)

@app.post("/predict")
def predict(data: ImageData):
    try:
        t1 = time.time()
        img_data = base64.b64decode(data.image)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        frame = np.array(img)
        s_img=copy.deepcopy(frame)
        processimg, class_result = score_img(frame, crop_task, target_task)
        print("class result",class_result)
        nut_count = len(class_result)
        if nut_count == 16 and len(set(class_result)) == 1:
            Status = "OK"
        else:
            Status = "NG"
            save_img(s_img)
        #processimg=cv2.resize(processimg,(800,800))
        processing_folder = r"G:\bolt_project\Seg_bolt_detection\processing"
        # Save processed image in the 'processing' folder
        timestamp = int(time.time() * 1000)
        output_filename = f"processed_image_{timestamp}.jpg"
        output_path = os.path.join(processing_folder, output_filename)
        #cv2.imwrite(output_path, processimg)
        _, img_encoded = cv2.imencode('.jpg', processimg)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        t2 = time.time()
        return {
            "image": img_base64,
            "Total_Nut": nut_count,
            "Status": Status,
            "Detected_Classes": class_result
        }
    except Exception as e:
        print(traceback.format_exc())
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return {
            "image": img_base64,
            "Total_Nut": 16,
            "Status": 'NG',
            "Detected_Classes": []
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=True)


############################YoLo Working Fine###############################

# import onnxruntime as ort
# import cv2
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import base64
# from PIL import Image
# from io import BytesIO
# import time
# import os
# import uvicorn
# from torchvision import transforms
# from ultralytics import YOLO  # for YOLOv8
# import traceback
# import torch

# # Define DetectionTask class
# class DetectionTask:
#     def __init__(self, model_path, class_file):
#         self.net = self.load_model(model_path)
#         self.classes = self.load_classes(class_file)

#     def load_model(self, model_path):
#         return cv2.dnn.readNetFromONNX(model_path)

#     def load_classes(self, class_file):
#         with open(class_file, 'r') as file:
#             return [line.strip() for line in file.readlines()]

#     # def perform_detection(self, frame, confidence_threshold=0.65, nms_threshold=0.1):
#     #     raise NotImplementedError("perform_detection method must be implemented in subclass.")

#     # def draw_boxes(self, frame, detections):
#     #     raise NotImplementedError("draw_boxes method must be implemented in subclass.")

# # Define TargetDetectionTask class using YOLOv8 for classification (Norm vs TVS)
# class TargetDetectionTask:
#     def __init__(self, confidence_threshold=0.5):
#         self.confidence_threshold = confidence_threshold
#         self.model = YOLO('best.pt')  # Load YOLOv8 model
#         self.model.to('cpu')  # Ensure model runs on CPU

#     def perform_classification(self, frame):

#         # Perform inference (no need for PIL conversion)
#         results = self.model.predict(frame)  

#         probs = getattr(results[0], 'probs', None)
#         if probs is not None:
#             probs = probs.data.cpu()  # Ensure tensor is on CPU
#             print("probs : ",probs)
#             max_idx = torch.argmax(probs).item()  # Get highest probability index
#             max_confidence = probs[max_idx].item()

#             # Ensure 'names' is correctly indexed
#             class_name = results[0].names.get(max_idx, "Unknown") if hasattr(results[0], 'names') else "Unknown"
#             return [(class_name, max_confidence)]
#         return [("Ok", 0.0)]

# # Define CropDetectionTask class
# class CropDetectionTask(DetectionTask):

#     def perform_detection(self, frame, confidence_threshold=0.6, nms_threshold=0.1):
#         if len(frame.shape) == 2:
#             height, width = frame.shape
#             channels = 1
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         else:
#             height, width, channels = frame.shape

#         blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
#         self.net.setInput(blob)
#         detections = self.net.forward()[0]

#         class_ids = []
#         confidences = []
#         boxes = []
#         x_scale, y_scale = width / 640, height / 640

#         for row in detections:
#             confidence = row[4]
#             if confidence > confidence_threshold:
#                 classes_score = row[5:]
#                 class_id = np.argmax(classes_score)
#                 if classes_score[class_id] > confidence_threshold:
#                     class_ids.append(class_id)
#                     confidences.append(confidence)
#                     cx, cy, w, h = row[:4]
#                     x1 = int((cx - w/2) * x_scale)
#                     y1 = int((cy - h/2) * y_scale)
#                     width = int(w * x_scale)
#                     height = int(h * y_scale)
#                     box = np.array([x1, y1, width, height])
#                     if width >= 150 and height >= 150:
#                         boxes.append(box)

#         # Ensure there are valid detections before applying NMS
#         if not boxes or not confidences:
#             print("No valid detections found.")
#             return []

#         indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

#         if len(indices) == 0:
#             print("No valid detections after NMS")
#             return []  # No valid detections after NMS

#         return [(boxes[i], self.classes[class_ids[i]], confidences[i]) for i in indices.flatten()]
#         # indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
#         # return [(boxes[i], self.classes[class_ids[i]], confidences[i]) for i in indices]
    

#     def draw_boxes(self, frame, crop_detections, target_task):
#         detected_labels = []
#         temp_color = (255, 255, 255)
#         results = []

#         for box, label, confidence in crop_detections:
#             x1, y1, w, h = box
#             roi = frame[y1:y1+h, x1:x1+w]
#             target_detections = target_task.perform_classification(roi)

#             for target_label, target_confidence in target_detections:
#                 detected_labels.append(target_label)
#                 results.append((box, target_label, temp_color))
#         print("detected_labels : ",detected_labels)
#         final_color = (0, 255, 0) if len(detected_labels) == 16 and len(set(detected_labels)) == 1 else (0, 0, 255)
#         for box, label, color in results:
#             x11, y11, w1, h1 = box
#             cv2.rectangle(frame, (x11, y11), (x11+w1, y11+h1), final_color, 4)
#             cv2.putText(frame, label, (x11, y11 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, final_color, 2, cv2.LINE_AA)

#         return frame, detected_labels

# app = FastAPI()

# class ImageData(BaseModel):
#     image: str

# @app.get("/ServerCheck")
# def read_root():
#     return {"status": "server is running"}

# def score_img(frame, crop_task, target_task):
#     crop_detections = crop_task.perform_detection(frame)
#     print("crop_detections : ",crop_detections)
#     return crop_task.draw_boxes(frame, crop_detections, target_task)

# crop_task = CropDetectionTask('det_bolt.onnx', 'crop.names')#gear_bolt
# target_task = TargetDetectionTask()
# output_folder = "G:/bolt_project/Seg_bolt_detection/processing"
# os.makedirs(output_folder, exist_ok=True)

# @app.post("/predict")
# def predict(data: ImageData):
#     try:
#         img_data = np.frombuffer(base64.b64decode(data.image), dtype=np.uint8)
#         frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
#         processed_img, class_result = score_img(frame, crop_task, target_task)
#         Status = "OK" if len(class_result) == 16 and len(set(class_result)) == 1 else "NG"
#         # # Save processed image in the 'processing' folder
#         # timestamp = int(time.time() * 1000)
#         # output_filename = f"processed_image_{timestamp}.jpg"
#         # output_path = os.path.join(output_folder, output_filename)
#         # cv2.imwrite(output_path, processed_img)
#         _, img_encoded = cv2.imencode('.jpg', processed_img)
#         return {"image": base64.b64encode(img_encoded).decode('utf-8'), "Total_Nut": len(class_result), "Status": Status, "Detected_Classes": class_result}
#     except Exception as e:
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(traceback.format_exc()))

# if __name__ == "__main__":
#     uvicorn.run('bolt_detection_yolo:app', host="127.0.0.1", port=5007, reload=True)



####################################working in production calssifier cnnn###################################################

# import onnxruntime as ort
# import cv2
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import base64
# from PIL import Image, ImageEnhance
# from io import BytesIO
# from PIL import Image
# import time
# import os
# import uvicorn
# from time import strftime
# from torchvision import transforms
# import torch
# from ultralytics import YOLO
# from collections import Counter
# import traceback

# # # Loading yolov8 classifer
# # model = YOLO('best.pt')  # Load YOLOv8 model
# # tempimg=cv2.imread("temp.png")
# # res=model.predict(tempimg)
# # print("temp done : ",res) #self.model.to('cpu')  # Ensure model runs on CPU


# # Path to the ONNX model
# onnx_model_path ='bolt_classifier_2025_old.onnx' #'tvs_norm_classifier_new.onnx'#
# # Load the ONNX model using onnxruntime and specify the CPU execution provider
# session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
# # Define the image transform (match the input preprocessing used for training)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),        # Resize image to 224x224
#     transforms.ToTensor(),                # Convert to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
# ])

# class DetectionTask:
#     def __init__(self, model_path, class_file):
#         self.net = self.load_model(model_path)
#         self.classes = self.load_classes(class_file)

#     def load_model(self, model_path):
#         return cv2.dnn.readNetFromONNX(model_path)

#     def load_classes(self, class_file):
#         with open(class_file, 'r') as file:
#             return [line.strip() for line in file.readlines()]
    

# class TargetDetectionTask:

#     def __init__(self, confidence_threshold=0.5):
#         self.confidence_threshold = confidence_threshold
#         self.model = YOLO('best.pt')  # Load YOLOv8 model
#         self.model.to('cpu')  # Ensure model runs on CPU

#     def ImageEnhancer(self,image,cls):

#         # Convert the result image to a PIL Image for further enhancement
#         result_pil_image = Image.fromarray(image)

#         # Adjust the brightness
#         if cls=="NORM":
#             #print("bright")
#             factor = 1.3  # fator greater then 1 increase brightness , lees then 1 decrease the brithness
#         else:
#             #print("dark")
#             factor = 0.7  # fator greater then 1 increase brightness , lees then 1 decrease the brithness
#         enhancer = ImageEnhance.Brightness(result_pil_image)
#         img_enhanced = enhancer.enhance(factor)
#         # Convert back to NumPy array if needed for further processing
#         proframe = np.array(img_enhanced)
#         return proframe

#     def find_idx_lower(self,res_list):
#         unique_values = set(res_list)
#         value_count={value:res_list.count(value) for value in unique_values}
#         lower_count = min(value_count, key=value_count.get)
#         idx_lower=[idx for idx, value in enumerate(res_list) if value == lower_count]
#         return idx_lower

#     def perform_classifier_yolo(self, frame):

#         # Perform inference (no need for PIL conversion)
#         results = self.model.predict(frame)  
#         #print("yolo results : ",results)
#         probs = getattr(results[0], 'probs', None)
#         if probs is not None:
#             probs = probs.data.cpu()  # Ensure tensor is on CPU
#             #print("probs : ",probs)
#             max_idx = torch.argmax(probs).item()  # Get highest probability index
#             max_confidence = probs[max_idx].item()

#             # Ensure 'names' is correctly indexed
#             class_name = results[0].names.get(max_idx, "Unknown") if hasattr(results[0], 'names') else "Unknown"
#             class_name = class_name.upper()
#             return [(class_name, max_confidence)]
#         return [("Ok", 0.0)]

#     def perform_classifier(self, frame):
#         image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
#         input_image = transform(image).unsqueeze(0).numpy()

#         input_name = session.get_inputs()[0].name
#         output_name = session.get_outputs()[0].name

#         outputs = session.run([output_name], {input_name: input_image})
#         print("outputs : ",outputs)
#         prediction = outputs[0]
#         print("prediction : ",prediction)

#         threshold = 3.5
#         if prediction[0] > threshold:
#             return "NORM"
#         else:
#             return "TVS"
        
#     def performaug_old(self, img, cls):
#         temp_res=[cls]
#         res=self.perform_classifier_yolo(img)
#         print("res : ",res)
#         temp_res.append(res)
#         inhanced_img=self.ImageEnhancer(img,cls)
#         res_inh=self.perform_classifier(inhanced_img)
#         print("res_inh : ",res_inh)
#         temp_res.append(res_inh)
#         counter=Counter(temp_res)
#         final_res, _ =counter.most_common(1)[0]

#         return final_res
    
#     def performaug(self, img, cls):
#         temp_res = [cls]  # cls is likely a string

#         res = self.perform_classifier_yolo(img)  # Returns a list of tuples
#         if res:
#             temp_res.append(res[0][0])  # Extract only the class name (string)

#         inhanced_img = self.ImageEnhancer(img, cls)
#         res_inh = self.perform_classifier(inhanced_img)  # This is already a string
#         temp_res.append(res_inh)
#         print("temp res : ",temp_res)
#         counter = Counter(temp_res)  # ✅ No more TypeError
#         final_res, _ = counter.most_common(1)[0]

#         return final_res

        

# class CropDetectionTask(DetectionTask):
#     def perform_detection(self, frame, confidence_threshold=0.65, nms_threshold=0.1):
#         if len(frame.shape) == 2:
#             height, width = frame.shape
#             channels = 1
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         else:
#             height, width, channels = frame.shape

#         blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
#         self.net.setInput(blob)
#         detections = self.net.forward()[0]

#         class_ids = []
#         confidences = []
#         boxes = []
#         x_scale, y_scale = width / 640, height / 640

#         for row in detections:
#             confidence = row[4]
#             if confidence > confidence_threshold:
#                 classes_score = row[5:]
#                 class_id = np.argmax(classes_score)
#                 if classes_score[class_id] > confidence_threshold:
#                     class_ids.append(class_id)
#                     confidences.append(confidence)
#                     cx, cy, w, h = row[:4]
#                     x1 = int((cx - w/2) * x_scale)
#                     y1 = int((cy - h/2) * y_scale)
#                     width = int(w * x_scale)
#                     height = int(h * y_scale)
#                     box = np.array([x1, y1, width, height])
#                     if width >= 150 and height >= 150:
#                         boxes.append(box)
#         # Ensure there are valid detections before applying NMS
#         if not boxes or not confidences:
#             print("No valid detections found.")
#             return []

#         indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
#         return [(boxes[i], self.classes[class_ids[i]], confidences[i]) for i in indices]


#     def draw_boxes(self, frame, crop_detections, target_task):
#         detected_labels = []

#         results = {}  # Store processed results for efficiency

#         for box, label, confidence in crop_detections:
#             x1, y1, w, h = box
#             roi = frame[y1:y1+h, x1:x1+w]

#             target_level = target_task.perform_classifier(roi)  # Single call
#             print("target_detection:", target_level)
#             #results.append((box, target_level))
#             results[(x1,y1,w,h)] = target_level
#             detected_labels.append(target_level)

#         try:

#             if len(set(detected_labels)) > 1:
#                 idx_lower=target_task.find_idx_lower(detected_labels)
#                 items_res=list(results.items())
#                 for idx in idx_lower:
#                     boxx,labN=items_res[idx]
#                     x11, y11, w1, h1=boxx
#                     imgA=frame[y11:y11+h1, x11:x11+w1]
#                     res_aug=target_task.performaug(imgA,labN)
#                     detected_labels[idx]=res_aug
#         except Exception as e:
#             print(traceback.format_exc())

#         # Determine final color based on detected labels
#         final_color = (0, 255, 0) if len(detected_labels) == 16 and len(set(detected_labels)) == 1 else (0, 0, 255)
#         i=0
#         # Draw results in a single loop
#         for box, label in results.items():
#             x1, y1, w, h = box
#             label=detected_labels[i]
#             i+=1
#             frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), final_color, 4)
#             frame = cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, final_color, 2, cv2.LINE_AA)

#         return frame, detected_labels


# app = FastAPI()

# class ImageData(BaseModel):
#     image: str

# @app.get("/ServerCheck")
# def read_root():
#     return {"status": "server is running"}

# def score_img(frame, crop_task, target_task):
#     crop_detections = crop_task.perform_detection(frame)
#     frame_with_detections, class_result = crop_task.draw_boxes(frame, crop_detections, target_task)

#     if not isinstance(frame_with_detections, np.ndarray):
#         raise ValueError("The processed image is not a valid NumPy array")

#     return frame_with_detections, class_result

# # load the model object
# crop_task = CropDetectionTask('gear_bolt.onnx', 'crop.names')
# target_task = TargetDetectionTask()

# @app.post("/predict")
# def predict(data: ImageData):
#     try:
#         t1 = time.time()
#         img_data = base64.b64decode(data.image)
#         img = Image.open(BytesIO(img_data)).convert('RGB')
#         frame = np.array(img)

#         processimg, class_result = score_img(frame, crop_task, target_task)
#         print("class result",class_result)
#         nut_count = len(class_result)
#         if nut_count == 16 and len(set(class_result)) == 1:
#             Status = "OK"
#         else:
#             Status = "NG"
#         #processimg=cv2.resize(processimg,(800,800))
#         processing_folder = r"G:\bolt_project\Seg_bolt_detection\processing"
#         # Save processed image in the 'processing' folder
#         timestamp = int(time.time() * 1000)
#         output_filename = f"processed_image_{timestamp}.jpg"
#         output_path = os.path.join(processing_folder, output_filename)
#         #cv2.imwrite(output_path, processimg)
#         _, img_encoded = cv2.imencode('.jpg', processimg)
#         img_base64 = base64.b64encode(img_encoded).decode('utf-8')
#         t2 = time.time()
#         return {
#             "image": img_base64,
#             "Total_Nut": nut_count,
#             "Status": Status,
#             "Detected_Classes": class_result
#         }
#     except Exception as e:
#         print(traceback.format_exc())
#         _, img_encoded = cv2.imencode('.jpg', frame)
#         img_base64 = base64.b64encode(img_encoded).decode('utf-8')
#         return {
#             "image": img_base64,
#             "Total_Nut": 16,
#             "Status": 'NG',
#             "Detected_Classes": []
#         }

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=5000, reload=True)
#########################################################################################


# # Convert RGB to grayscale for CLAHE processing
# gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

# # Create a mask for overexposed areas (pixel values > 245)
# mask = np.where(gray_image > 245, 255, 0).astype(np.uint8)

# # Initialize CLAHE
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# # Apply CLAHE to the grayscale image
# clahe_image = clahe.apply(gray_image)

# # Combine the original grayscale image with the CLAHE-enhanced image
# result_image = np.where(mask == 255, gray_image, clahe_image)

# # Convert the result image back to RGB to match the original image's format
# result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)

# # Convert the result image to a PIL Image for further enhancement
# result_pil_image = Image.fromarray(result_image_rgb)

# # Adjust the brightness
# factor = 0.8  # Brightness adjustment factor
# enhancer = ImageEnhance.Brightness(result_pil_image)
# img_enhanced = enhancer.enhance(factor)
# # Convert back to NumPy array if needed for further processing
# proframe = np.array(img_enhanced)


#################################################################

