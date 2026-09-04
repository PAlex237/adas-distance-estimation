import cv2
import numpy as np
from ultralytics import YOLO

# Parametri de configurare
BEV_LENGTH_METERS = 25.0 
WARNING_DISTANCE = 2.0 
ORIZONT_Y = 0.65  
OFFSET_X = 0.029  
LATIME_SUS = 0.05
LATIME_JOS = 0.40

class DistanceKalmanFilter:
    """Clasa pentru netezirea distanței folosind filtrul Kalman."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(2, 1) # Stare: [distanta, viteza]
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf.processNoiseCov = np.array([[1e-4, 0], [0, 1e-4]], np.float32)
        self.kf.measurementNoiseCov = np.array([[1e-1]], np.float32)
        
    def update(self, measurement):
        self.kf.predict() # Estimează poziția următoare
        self.kf.correct(np.array([[np.float32(measurement)]])) # Corectează cu detecția YOLO
        return self.kf.statePost[0][0]

def main():
    model = YOLO('yolov8n.pt') # Modelul de detecție YOLOv8
    cap = cv2.VideoCapture('data/video_test.mp4')

    # Obține rezoluția video-ului
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # Geometria pentru "Bird's Eye View" (Homografie)
    center_x = 0.5 + OFFSET_X
    src_pts = np.float32([
        [w * (center_x - LATIME_SUS / 2), h * ORIZONT_Y], 
        [w * (center_x + LATIME_SUS / 2), h * ORIZONT_Y], 
        [w * (center_x + LATIME_JOS / 2), h],             
        [w * (center_x - LATIME_JOS / 2), h]              
    ])
    bev_w, bev_h = 400, 600
    dst_pts = np.float32([[0, 0], [bev_w, 0], [bev_w, bev_h], [0, bev_h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    meters_per_pixel_y = BEV_LENGTH_METERS / bev_h

    lead_vehicle_kf = DistanceKalmanFilter()
    kf_initialized = False 

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Vizualizare zonă de interes (Trapez) și transformare Bird's Eye View
        cv2.polylines(frame, [src_pts.astype(np.int32).reshape((-1, 1, 2))], True, (255, 0, 0), 2)
        bev_frame = cv2.warpPerspective(frame, matrix, (bev_w, bev_h))
        
        # 2. Detecție obiecte YOLO
        results = model(frame, classes=[2, 5, 7]) 
        result = results[0]
        
        closest_distance = float('inf')
        lead_box = None
        lead_point_bev = None
        
        # 3. Identificarea vehiculului țintă (Lead Vehicle)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if y2 >= h * 0.90: continue # Ignoră capota mașinii
            
            bottom_center = (int((x1 + x2) / 2), min(y2, h - 1))
            is_in_lane = cv2.pointPolygonTest(src_pts.astype(np.int32).reshape((-1, 1, 2)), bottom_center, False) >= 0
            
            if is_in_lane:
                # 4. Calcul distanței folosind proiecția BEV
                point_bev = cv2.perspectiveTransform(np.array([[[bottom_center[0], bottom_center[1]]]], dtype=np.float32), matrix)
                distance_meters = (bev_h - point_bev[0][0][1]) * meters_per_pixel_y
                
                if distance_meters < closest_distance:
                    closest_distance = distance_meters
                    lead_box = (x1, y1, x2, y2)
                    lead_point_bev = point_bev
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 150), 1)
        
        # 5. Stabilizare Kalman și Alertare
        if lead_box is not None:
            if not kf_initialized:
                lead_vehicle_kf.kf.statePost = np.array([[np.float32(closest_distance)], [0]], np.float32)
                kf_initialized = True
            
            smoothed_dist = max(0.0, lead_vehicle_kf.update(closest_distance))
            
            # AICI ESTE LOGICA DE ALERTĂ VERIFICATĂ:
            if smoothed_dist < WARNING_DISTANCE:
                color = (0, 0, 255) # Roșu pentru alertă
                # Desenăm textul de alertă direct pe frame
                cv2.putText(frame, f"!!! ALERTA: DISTANTA MICA ({smoothed_dist:.1f}m) !!!", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            else:
                color = (0, 255, 0) # Verde dacă e totul în regulă
            
            # Desenăm chenarul și distanța
            cv2.rectangle(frame, (lead_box[0], lead_box[1]), (lead_box[2], lead_box[3]), color, 2)
            cv2.putText(frame, f"Dist: {smoothed_dist:.1f} m", (lead_box[0], lead_box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            bev_x, bev_y = int(lead_point_bev[0][0][0]), int(lead_point_bev[0][0][1])
            if 0 <= bev_x < bev_w and 0 <= bev_y < bev_h:
                cv2.circle(bev_frame, (bev_x, bev_y), 6, color, -1)

        cv2.imshow('Distance Estimation to Lead Vehicle', frame)
        cv2.imshow('Bird Eye View', bev_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()