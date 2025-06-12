import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import numpy as np
import cv2
import os
import base64

def get_input(frame):

    indices=[0, 1, 2, 4, 5, 6, 7, 10, 13, 14, 17, 19, 21, 33, 37, 39, 40, 45, 48, 54, 58, 61, 64, 67, 78, 80, 81, 82, 84, 87, 88, 91, 93, 94, 95, 97, 98, 103, 109, 115, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 220, 234, 246, 249, 251, 263, 267, 269, 270, 275, 278, 284, 288, 291, 294, 297, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 326, 327, 332, 338, 344, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 440, 454, 466]
    face_detection = mp.solutions.face_detection.FaceDetection()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    t_some=[]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bounding_box = detection.location_data.relative_bounding_box
            x = int(bounding_box.xmin * frame.shape[1])
            y = int(bounding_box.ymin * frame.shape[0])
            w = int(bounding_box.width * frame.shape[1])
            h = int(bounding_box.height * frame.shape[0])
    else:
        return False

    crop_img = frame[y-30:y+h+10, x-20:x+w+20]

    rgb_frame = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for i in results.multi_face_landmarks:
            for j in indices:
                t_some.append(i.landmark[j].x)
                t_some.append(i.landmark[j].y)
    else:
        return False
                
    return t_some, crop_img

app = FastAPI()


@app.get("/")
async def welcome():
    return {"messaage":"Welcome!"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), model_name : str = None):       #inputs = [Decissiontree, MLR, Randomforest, SVM]
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    o = get_input(img)
    if not o:
        raise HTTPException(status_code=404, detail="Detection failed")
    t_some, crop_img = o

    if not model_name:
        raise HTTPException(status_code=404, detail="Model not selected")
    else:
        modle_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Degree regression model\\models\\"+model_name+"_tilt.joblib")
        model_tilt = joblib.load(modle_path)
        modle_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Degree regression model\\models\\"+model_name+"_turn.joblib")
        model_turn = joblib.load(modle_path)
        modle_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Degree regression model\\models\\"+model_name+"_ud.joblib")
        model_ud = joblib.load(modle_path)
    degree_tilt = model_tilt.predict([t_some])[0]
    degree_turn = model_turn.predict([t_some])[0]
    degree_ud = model_ud.predict([t_some])[0]
    _, encoded_img = cv2.imencode(".png", crop_img)
    img_base64 = base64.b64encode(encoded_img).decode('utf-8')

    return {"degree_tilt": degree_tilt, "degree_turn": degree_turn, "degree_ud": degree_ud, "image": img_base64}



