import cv2
import mediapipe as mp

degree=0
listin_some=[]
listin_all=[]

#indices of the points of interest
indices=[0, 1, 2, 4, 5, 6, 7, 10, 13, 14, 17, 19, 21, 33, 37, 39, 40, 45, 48, 54, 58, 61, 64, 67, 78, 80, 81, 82, 84, 87, 88, 91, 93, 94, 95, 97, 98, 103, 109, 115, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 220, 234, 246, 249, 251, 263, 267, 269, 270, 275, 278, 284, 288, 291, 294, 297, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 326, 327, 332, 338, 344, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 440, 454, 466]

cap=cv2.VideoCapture(0)
face_detection = mp.solutions.face_detection.FaceDetection()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


left_end=0.55
right_end=0.48
right_rest=0.25
left_rest=0.75
degree=-9999
nose_norm=-1
left_norm=-1
right_norm=-1
left_ind=263
right_ind=33

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("camera not working")
        exit()

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
        continue
    crop_img = frame[y-30:y+h+10, x-20:x+w+20]

    rgb_frame = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for i in results.multi_face_landmarks:
            point=(int(i.landmark[1].x*crop_img.shape[1]),int(i.landmark[1].y*crop_img.shape[0]))
            cv2.circle(crop_img, point, radius=5, color=(0, 0, 255), thickness=-1)
            nose_norm=round(i.landmark[1].x,2)
            left_point=(int(i.landmark[left_ind].x*crop_img.shape[1]),int(i.landmark[left_ind].y*crop_img.shape[0]))
            cv2.circle(crop_img, left_point, radius=5, color=(0, 0, 255), thickness=-1)
            left_norm=round(i.landmark[left_ind].x,2)
            right_point=(int(i.landmark[right_ind].x*crop_img.shape[1]),int(i.landmark[right_ind].y*crop_img.shape[0]))
            cv2.circle(crop_img, right_point, radius=5, color=(0, 0, 255), thickness=-1)
            right_norm=round(i.landmark[right_ind].x,2)

            #print(left_norm,",",nose_norm,",",right_norm)

            t_some=[]
            #some
            for j in indices:
                t_some.append(i.landmark[j].x)
                t_some.append(i.landmark[j].y)
                cv2.circle(crop_img, (int(i.landmark[j].x*crop_img.shape[1]),int(i.landmark[j].y*crop_img.shape[0])), radius=5, color=(0, 0, 255), thickness=-1)
            t_all=[]
            #all
            for j in range(len(i.landmark)):
                t_all.append(i.landmark[j].x)
                t_all.append(i.landmark[j].y)
    else:
        print("Feature detection failed")
        continue
        

    mirrored_frame = cv2.flip(crop_img, 1)

    if nose_norm<0.5:
        degree=-int(((((left_norm-right_end)/(left_rest-right_end))*90)-90))
        #cv2.putText(mirrored_frame, f"{degree:.2f}", (0,mirrored_frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(mirrored_frame, str(degree), (0,mirrored_frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        degree=int((((left_end-right_norm)/(left_end-right_rest))*90)-90)
        #cv2.putText(mirrored_frame, f"{degree:.2f}", (0,mirrored_frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(mirrored_frame, str(degree), (0,mirrored_frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    listin_some.append(t_some+[degree])
    listin_all.append(t_all+[degree])
    
    cv2.imshow("Cropped Image", mirrored_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break  
    
cap.release()
cv2.destroyAllWindows()

for i in listin_some:
    pass
    #print(i[-1])
print(len(listin_some))

listin_some.insert(0, list(range(1,265))+["degree"])
if input()=='s':
    pass
    '''with open("turn_train_crop_some.csv",'a',newline="") as f:
        writer=csv.writer(f)
        writer.writerows(listin_some)
    with open("turn_train_crop_all.csv",'a',newline="") as f:
        writer=csv.writer(f)
        writer.writerows(listin_all)'''
    
    '''with open("turn_test_crop_some.csv",'a',newline="") as f:
        writer=csv.writer(f)
        writer.writerows(listin_some)
    with open("turn_test_crop_all.csv",'a',newline="") as f:
        writer=csv.writer(f)
        writer.writerows(listin_all)'''