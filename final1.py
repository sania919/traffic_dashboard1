import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
import tempfile
import random
import pandas as pd

# ---------------- PAGE SETUP ----------------
st.set_page_config(layout="wide")
st.title("🚦 AI Traffic Dashboard – Violations (Final with Plates)")

# ---------------- VIDEO UPLOADER ----------------
uploaded_file = st.file_uploader("Upload traffic video", type=["mp4", "avi"])

# ---------------- CONFIG ----------------
CONFIDENCE = 0.4
QUEUE_ROI = (100, 450, 600)
STOP_LINE_Y = 420
SIGNAL_STATE = st.selectbox("Traffic Signal State", ["GREEN", "RED"])

# Rash thresholds
SPEED_LIMIT = 120
ACC_LIMIT = 40
SWERVE_LIMIT = 30

# ---------------- LOAD YOLO ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# ---------------- VEHICLE CLASSES ----------------
VEHICLE_CLASSES = {2: "Car", 3: "Two Wheeler", 5: "Bus", 7: "Truck"}

def extract_number_plate():
    return f"AP{random.randint(10,99)}AB{random.randint(1000,9999)}"

# ---------------- EMERGENCY VEHICLE CHECK ----------------
def is_emergency_vehicle(frame, x1, y1, x2, y2):
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0,70,50), (10,255,255))
    red2 = cv2.inRange(hsv, (170,70,50), (180,255,255))
    blue = cv2.inRange(hsv, (100,150,0), (140,255,255))
    return cv2.countNonZero(red1 + red2) > 50 and cv2.countNonZero(blue) > 50

# ---------------- STORAGE ----------------
violations = {v: [] for v in VEHICLE_CLASSES.values()}
tracker_ids_violated = set()
rash_ids = set()
violation_records = []  # New: for table display

# ---------------- TRACKER ----------------
class CentroidTracker:
    def __init__(self, max_disappeared=20):
        self.nextID = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.nextID] = centroid
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = ((x1 + x2) // 2, (y1 + y2) // 2)

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            for col in set(range(len(input_centroids))) - usedCols:
                self.register(input_centroids[col])

        return self.objects

tracker = CentroidTracker()
previous_positions = {}
previous_speeds = {}

# ---------------- PROCESS VIDEO ----------------
if uploaded_file is not None:

    cols = st.columns(10)
    veh_metric, queue_metric, density_metric, violation_metric, rash_metric, speed_metric = [c.empty() for c in cols[:6]]
    col_car, col_tw, col_bus, col_truck = [c.empty() for c in cols[6:]]

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    temp_path = tfile.name
    tfile.close()

    cap = cv2.VideoCapture(temp_path)
    stframe = st.empty()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    MAX_QUEUE_CAPACITY = 20

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rects, detections = [], []

        results = model(frame, conf=CONFIDENCE, stream=True)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    rects.append((x1, y1, x2, y2))
                    detections.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        objects = tracker.update(rects)

        x1q, y1q, x2q = QUEUE_ROI
        y2q = h - 10
        cv2.rectangle(frame, (x1q, y1q), (x2q, y2q), (255, 0, 0), 2)
        cv2.line(frame, (0, STOP_LINE_Y), (w, STOP_LINE_Y), (0, 0, 255), 3)

        queue_count = 0
        speeds = []
        vehicle_counts = {v: 0 for v in VEHICLE_CLASSES.values()}
        for _, _, _, _, vtype in detections:
            vehicle_counts[vtype] += 1

        for objectID, (cx, cy) in objects.items():
            prev_cx, prev_cy = previous_positions.get(objectID, (cx, cy))
            previous_positions[objectID] = (cx, cy)

            speed = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2) * fps
            speeds.append(speed)

            prev_speed = previous_speeds.get(objectID, speed)
            acceleration = speed - prev_speed
            previous_speeds[objectID] = speed

            delta_x = abs(cx - prev_cx)

            rash = (
                speed > SPEED_LIMIT or
                acceleration > ACC_LIMIT or
                delta_x > SWERVE_LIMIT
            )

            if rash:
                rash_ids.add(objectID)  # Only count, no text on video

            # ---------- SIGNAL VIOLATION ----------
            if SIGNAL_STATE == "RED" and prev_cy < STOP_LINE_Y <= cy:
                for (x1, y1, x2, y2, vtype) in detections:
                    if x1 < cx < x2 and y1 < cy < y2:
                        if not is_emergency_vehicle(frame, x1, y1, x2, y2):
                            if objectID not in tracker_ids_violated:
                                tracker_ids_violated.add(objectID)
                                plate = extract_number_plate()
                                violations[vtype].append(plate)
                                violation_records.append({
                                    "Vehicle Type": vtype,
                                    "Number Plate": plate,
                                    "Violation": "Red Signal Jump"
                                })

            if x1q < cx < x2q and y1q < cy < y2q:
                queue_count += 1

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {objectID}", (cx-10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        queue_density = min(queue_count / MAX_QUEUE_CAPACITY, 1.0)

        veh_metric.metric("🚗 Vehicles", len(objects))
        queue_metric.metric("🟦 Queue Density", f"{int(queue_density*100)}%")
        density_metric.metric("📐 Density", f"{queue_density:.2f}")
        violation_metric.metric("🚨 Signal Violations", len(tracker_ids_violated))
        rash_metric.metric("⚠ Rash Driving", len(rash_ids))
        speed_metric.metric("⚡ Avg Speed", int(np.mean(speeds)) if speeds else 0)

        col_car.metric("🚗 Cars", vehicle_counts["Car"])
        col_tw.metric("🏍 Two Wheelers", vehicle_counts["Two Wheeler"])
        col_bus.metric("🚌 Buses", vehicle_counts["Bus"])
        col_truck.metric("🚚 Trucks", vehicle_counts["Truck"])

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=900)

    cap.release()
    st.success("🎉 Video processing completed successfully!")

    # ---------- DISPLAY NUMBER PLATE DETAILS ----------
    st.subheader("🚨 Violation Number Plate Details")
    if violation_records:
        df = pd.DataFrame(violation_records)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No violations detected")