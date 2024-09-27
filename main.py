import cv2
import torch
import time
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
#import RPi.GPIO as GPIO

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

client = MongoClient('Insert MongoDB String here')
db = client['traffic_data']
collection = db['time_series']

def detect_objects(frame):
    results = model(frame)
    vehicles = 0

    for result in results.xyxy[0]:  # each detection
        if int(result[5]) in [2, 3, 5, 7]: # 2: car, 3: motorcycle, 5: bus, 7: truck (vehicle classes)
            vehicles += 1

    return vehicles


cap = cv2.VideoCapture('video_file_name.mp4')  

threshold = 5 
distance_threshold = 50 #have to implement this in frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    vehicles_count = detect_objects(frame)
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    record = {'timestamp': timestamp, 'vehicle_count': vehicles_count}
    collection.insert_one(record)

    cv2.putText(frame, f'Vehicles: {vehicles_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Traffic Monitoring', frame)

    if vehicles_count > threshold:
        print("Congestion detected!")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

data = list(collection.find({}, {'_id': 0, 'timestamp': 1, 'vehicle_count': 1}))
df = pd.DataFrame(data)

#timestamp to numeric value for clustering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

#K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(df[['hour', 'vehicle_count']])

#cluster labels
df['cluster'] = kmeans.labels_

#K-Means Clustering plot have to figure this out
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='hour', y='vehicle_count', hue='cluster', palette='deep', s=100)
plt.title("K-Means Clustering of Traffic Congestion Over Time", fontsize=16)
plt.xlabel("Hour of the Day", fontsize=14)
plt.ylabel("Vehicle Count", fontsize=14)
plt.legend(title='Cluster')
plt.grid(True)
plt.show(block=True)

#top 3 most congested times
top_clusters = df.groupby('cluster').mean().sort_values(by='vehicle_count', ascending=False)
print("Top 3 most congested hours:")
print(top_clusters.head(3))

"""
#traffic light with GPIO
RED_PIN = 17
YELLOW_PIN = 27
GREEN_PIN = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)

def gpio_traffic_light(vehicle_count):
    if vehicle_count > threshold:
        GPIO.output(RED_PIN, GPIO.HIGH)
        GPIO.output(YELLOW_PIN, GPIO.LOW)
        GPIO.output(GREEN_PIN, GPIO.LOW)
    elif vehicle_count > (threshold // 2):
        GPIO.output(RED_PIN, GPIO.LOW)
        GPIO.output(YELLOW_PIN, GPIO.HIGH)
        GPIO.output(GREEN_PIN, GPIO.LOW)
    else:
        GPIO.output(RED_PIN, GPIO.LOW)
        GPIO.output(YELLOW_PIN, GPIO.LOW)
        GPIO.output(GREEN_PIN, GPIO.HIGH)

try:
    while True:
        latest_data = collection.find_one(sort=[('_id', -1)])
        if latest_data:
            gpio_traffic_light(latest_data['vehicle_count'])
        time.sleep(1)
finally:
    GPIO.cleanup()
"""
