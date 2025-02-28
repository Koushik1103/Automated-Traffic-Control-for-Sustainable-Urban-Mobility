from flask import Flask, jsonify
import cv2
import numpy as np
import time
import base64
from ultralytics import YOLO
from geopy.geocoders import Nominatim
from pymongo import MongoClient
from sklearn.cluster import KMeans
from threading import Thread
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n.pt")

client = MongoClient("mongodb+srv://koushik_new:automatedtrafficcontrol@cluster0.j9bokzd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["traffic_management"]
collection = db["vehicle_counts"]

COLORS = {
    'person': (255, 255, 255),
    'car': (255, 0, 0),
    'motorbike': (0, 255, 0),
    'truck': (0, 255, 255),
    'bus': (255, 255, 0),
    'ambulance': (0, 0, 255),
    'firetruck': (0, 0, 255)
}

geoLoc = Nominatim(user_agent="GetLoc")
locname = geoLoc.reverse("12.953195, 80.141602")
location_address = locname.address if locname else "Unknown"

latest_frames = {"direction_1": None, "direction_2": None}
traffic_signal = {"direction_1": "RED", "direction_2": "RED", "pedestrian": "RED"}

def process_frames():
    global latest_frames, traffic_signal
    cap1 = cv2.VideoCapture('carsdemo.mp4')  
    cap2 = cv2.VideoCapture('carsdemo1.mp4') 
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: One or both cameras could not be opened!")
        return
    
    cycle_count = 0
    
    while True:
        vehicle_counts = {"direction_1": 0, "direction_2": 0}
        
        for direction, cap in zip(["direction_1", "direction_2"], [cap1, cap2]):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame for {direction}")
                continue
            
            results = model(frame)
            vehicle_count = {key: 0 for key in COLORS.keys()}
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name in COLORS:
                        vehicle_count[class_name] += 1
                        color = COLORS[class_name]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, class_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            latest_frames[direction] = frame.copy()
            vehicle_counts[direction] = sum(vehicle_count.values())
            
            document = {
                "direction": direction,
                "vehicle_counts": vehicle_count,
                "total_count": vehicle_counts[direction],
                "timestamp": int(time.time() * 1000),
                "location": location_address
            }
            collection.insert_one(document)
            print(f"Data updated in MongoDB for {direction}: {vehicle_counts[direction]} vehicles")
        
        
        if cycle_count % 3 == 2:  
            traffic_signal = {"direction_1": "RED", "direction_2": "RED", "pedestrian": "GREEN"}
            print("🚦 Pedestrian signal activated for 60s")
            time.sleep(60)
        else:
            if vehicle_counts["direction_1"] > vehicle_counts["direction_2"] * 1.5:
                traffic_signal = {"direction_1": "GREEN", "direction_2": "RED", "pedestrian": "RED"}
                green_time = min(60, max(20, vehicle_counts["direction_1"] // 5))
                print(f"🚦 Direction 1 Green for {green_time}s")
                time.sleep(green_time)
            elif vehicle_counts["direction_2"] > vehicle_counts["direction_1"] * 1.5:
                traffic_signal = {"direction_1": "RED", "direction_2": "GREEN", "pedestrian": "RED"}
                green_time = min(60, max(20, vehicle_counts["direction_2"] // 5))
                print(f"🚦 Direction 2 Green for {green_time}s")
                time.sleep(green_time)
            else:
                traffic_signal = {"direction_1": "GREEN", "direction_2": "RED", "pedestrian": "RED"}
                print("🚦 Direction 1 Green for 30s")
                time.sleep(30)
                traffic_signal = {"direction_1": "RED", "direction_2": "GREEN", "pedestrian": "RED"}
                print("🚦 Direction 2 Green for 30s")
                time.sleep(30)
        
        cycle_count += 1
        time.sleep(20)


def get_top_congested_locations():
    data = list(collection.find({}, {"_id": 0, "total_count": 1, "location": 1}))
    
    if len(data) < 3:
        return {"error": "Not enough data points for clustering."}
    
    vehicle_counts = np.array([int(d["total_count"]) for d in data]).reshape(-1, 1)
    locations = [d["location"] for d in data]
    
    unique_counts = np.unique(vehicle_counts)
    if len(unique_counts) < 3:
        return sorted(data, key=lambda x: x["total_count"], reverse=True)[:3]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vehicle_counts)
    
    return sorted([{ "location": loc, "congestion_score": int(count) } for loc, count in zip(locations, vehicle_counts.flatten())], key=lambda x: x["congestion_score"], reverse=True)[:3]


@app.route('/get_traffic_data', methods=['GET'])
def get_traffic_data():
    encoded_images = {direction: encode_frame_to_base64(direction) for direction in latest_frames}
    top_congested = get_top_congested_locations()
    return jsonify({
        "latest_frames": encoded_images,
        "traffic_signal": traffic_signal,
        "top_3_congested_locations": top_congested
    })


def encode_frame_to_base64(direction):
    if latest_frames[direction] is None:
        return None
    _, buffer = cv2.imencode('.jpg', latest_frames[direction])
    return base64.b64encode(buffer).decode("utf-8")

if __name__ == '__main__':
    Thread(target=process_frames, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", port=5000)

# from flask import Flask, jsonify
# import cv2
# import numpy as np
# import time
# import base64
# from ultralytics import YOLO
# from geopy.geocoders import Nominatim
# from pymongo import MongoClient
# from sklearn.cluster import KMeans
# from threading import Thread

# app = Flask(__name__)

# # Initialize YOLO model
# model = YOLO("yolov8n.pt")

# # MongoDB Connection
# client = MongoClient("mongodb+srv://koushik_new:automatedtrafficcontrol@cluster0.j9bokzd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# db = client["traffic_management"]
# collection = db["vehicle_counts"]

# # Color Mapping for Objects
# COLORS = {
#     'person': (255, 255, 255),
#     'car': (255, 0, 0),
#     'motorbike': (0, 255, 0),
#     'truck': (0, 255, 255),
#     'bus': (255, 255, 0),
#     'ambulance': (0, 0, 255),
#     'firetruck': (0, 0, 255)
# }

# # Get Location Details
# geoLoc = Nominatim(user_agent="GetLoc")
# locname = geoLoc.reverse("12.953195, 80.141602")
# location_address = locname.address if locname else "Unknown"

# latest_frames = {"direction_1": None, "direction_2": None}
# traffic_signal = {"direction_1": "RED", "direction_2": "RED", "pedestrian": "RED"}

# # Function to process frames every 20 seconds
# def process_frames():
#     global latest_frames, traffic_signal
#     cap1 = cv2.VideoCapture('test/amb.jpg')  # Camera for one direction
#     cap2 = cv2.VideoCapture('test/photo-1584045528666-8279dfe6a9ec.jpeg')  # Camera for opposite direction
#     cycle_count = 0
    
#     while True:
#         vehicle_counts = {}
#         for direction, cap in zip(["direction_1", "direction_2"], [cap1, cap2]):
#             ret, frame = cap.read()
#             if not ret:
#                 continue
            
#             vehicle_count = {key: 0 for key in COLORS.keys()}
#             results = model(frame)
            
#             for result in results:
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = box.xyxy[0].tolist()
#                     class_id = int(box.cls[0])
#                     class_name = model.names[class_id]
#                     if class_name in COLORS:
#                         vehicle_count[class_name] += 1
#                         color = COLORS[class_name]
#                         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                         cv2.putText(frame, class_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
#             latest_frames[direction] = frame.copy()
#             vehicle_counts[direction] = sum(vehicle_count.values())
            
#             document = {
#                 "direction": direction,
#                 "vehicle_counts": vehicle_count,
#                 "total_count": vehicle_counts[direction],
#                 "timestamp": int(time.time() * 1000),
#                 "location": location_address
#             }
#             collection.insert_one(document)
#             print(f"Data updated in MongoDB for {direction}")
        
#         # Traffic Signal Algorithm
#         if cycle_count % 3 == 2:  # Every two signal cycles, give pedestrian crossing time
#             traffic_signal = {"direction_1": "RED", "direction_2": "RED", "pedestrian": "GREEN"}
#             time.sleep(60)
#         else:
#             if vehicle_counts["direction_1"] > vehicle_counts["direction_2"] * 1.5:
#                 traffic_signal = {"direction_1": "GREEN", "direction_2": "RED", "pedestrian": "RED"}
#                 time.sleep(min(60, max(20, vehicle_counts["direction_1"] // 5)))
#             elif vehicle_counts["direction_2"] > vehicle_counts["direction_1"] * 1.5:
#                 traffic_signal = {"direction_1": "RED", "direction_2": "GREEN", "pedestrian": "RED"}
#                 time.sleep(min(60, max(20, vehicle_counts["direction_2"] // 5)))
#             else:
#                 traffic_signal = {"direction_1": "GREEN", "direction_2": "RED", "pedestrian": "RED"}
#                 time.sleep(30)
#                 traffic_signal = {"direction_1": "RED", "direction_2": "GREEN", "pedestrian": "RED"}
#                 time.sleep(30)
        
#         cycle_count += 1
#         time.sleep(20)

# # Function to get top 3 congested locations
# def get_top_congested_locations():
#     data = list(collection.find({}, {"_id": 0, "total_count": 1, "location": 1}))
    
#     if len(data) < 3:
#         return {"error": "Not enough data points for clustering."}
    
#     vehicle_counts = np.array([int(d["total_count"]) for d in data]).reshape(-1, 1)
#     locations = [d["location"] for d in data]
    
#     unique_counts = np.unique(vehicle_counts)
#     if len(unique_counts) < 3:
#         return sorted(data, key=lambda x: x["total_count"], reverse=True)[:3]
    
#     kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(vehicle_counts)
    
#     return sorted([{ "location": loc, "congestion_score": int(count) } for loc, count in zip(locations, vehicle_counts.flatten())], key=lambda x: x["congestion_score"], reverse=True)[:3]

# # API Endpoint to Get Latest Data
# @app.route('/get_traffic_data', methods=['GET'])
# def get_traffic_data():
#     encoded_images = {direction: encode_frame_to_base64(direction) for direction in latest_frames}
#     top_congested = get_top_congested_locations()
#     return jsonify({
#         "latest_frames": encoded_images,
#         "traffic_signal": traffic_signal,
#         "top_3_congested_locations": top_congested
#     })

# # Function to encode frames as base64 images
# def encode_frame_to_base64(direction):
#     if latest_frames[direction] is None:
#         return None
#     _, buffer = cv2.imencode('.jpg', latest_frames[direction])
#     return base64.b64encode(buffer).decode("utf-8")

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=5000)
