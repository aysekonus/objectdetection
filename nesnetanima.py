import cv2
import numpy as np

# import img
img = cv2.imread("C:/Users/Lenovo/Desktop/YOLO/besiktas-barbaros-bulvarinda-kirmizi-isikta-donus-yapan-surucunun-otomobiline-yokus-asagiya-seyir-halinde-olan-motosikletli-c-IHA-20171030AW211318-4-t.jpg")

# img settings
img_width = img.shape[1]
img_height = img.shape[0]
img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB = True)

# import yolo coco data set
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack"
          ]

# color settings
colors = ["255,255,0","0,255,0","255,0,255","0,255,255","0,0,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(20,1))


# import yolo model
model = cv2.dnn.readNetFromDarknet("C:/Users/Lenovo/Desktop/YOLO/MODEL/yolov3.cfg","C:/Users/Lenovo/Desktop/YOLO/MODEL/yolov3.weights")
layers = model.getLayerNames()
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)

ids_list = []
boxes_list = []
confidences_list = []


for detection_layers in detection_layers:
    for object_detection in detection_layers:
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.7:
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            
            start_x = int(box_center_x - (box_width / 2))
            start_y = int(box_center_y - (box_height / 2))
            
            
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
            
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
            
for max_id in max_ids:
    max_class_id = max_id[0]
    box = boxes_list[max_class_id]
                
    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]
                
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]
            
            
    end_x = start_x + box_width
    end_y = start_y + box_height
            
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]
            
    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,2)
    cv2.putText(img,label,(start_x,start_y-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)

            

while True:
    cv2.imshow("Tespit Ekrani", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        img = cv2.imread(img)
            



