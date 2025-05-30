import sys
import setuptools
sys.modules['distutils'] = setuptools

import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

def main():
    def las_draw_fancy_box(las_img, las_pt1, las_pt2, las_color, las_thickness):
        las_x1, las_y1 = int(las_pt1[0]), int(las_pt1[1])
        las_x2, las_y2 = int(las_pt2[0]), int(las_pt2[1])
        las_line_length = int((las_x2 - las_x1) * 0.2)
        cv2.line(las_img, (las_x1, las_y1), (las_x1+las_line_length, las_y1), las_color, las_thickness)
        cv2.line(las_img, (las_x1, las_y1), (las_x1, las_y1+las_line_length), las_color, las_thickness)
        cv2.line(las_img, (las_x2, las_y1), (las_x2-las_line_length, las_y1), las_color, las_thickness)
        cv2.line(las_img, (las_x2, las_y1), (las_x2, las_y1+las_line_length), las_color, las_thickness)
        cv2.line(las_img, (las_x1, las_y2), (las_x1+las_line_length, las_y2), las_color, las_thickness)
        cv2.line(las_img, (las_x1, las_y2), (las_x1, las_y2-las_line_length), las_color, las_thickness)
        cv2.line(las_img, (las_x2, las_y2), (las_x2-las_line_length, las_y2), las_color, las_thickness)
        cv2.line(las_img, (las_x2, las_y2), (las_x2, las_y2-las_line_length), las_color, las_thickness)

    las_model = YOLO(r"C:\1\v8.pt")
    las_tracker = DeepSort(max_age=30)
    las_person_records = {}
    las_id_mapping = {}
    las_id_counter = 0
    las_scroll_offset = 0
    las_cap = cv2.VideoCapture(0)
    cv2.namedWindow("Rona Vision AI", cv2.WINDOW_NORMAL)
    def nothing(las_x):
        pass
    cv2.createTrackbar("Scroll", "Rona Vision AI", 0, 100, nothing)
    
    while las_cap.isOpened():
        las_ret, las_frame = las_cap.read()
        if not las_ret:
            break
        las_results = las_model.predict(las_frame, conf=0.4, classes=[0])[0]
        las_detections = []
        for las_box in las_results.boxes.data.tolist():
            las_x1, las_y1, las_x2, las_y2, las_score, las_class_id = las_box
            las_detections.append(([las_x1, las_y1, las_x2 - las_x1, las_y2 - las_y1], las_score, 'person'))
        print(f"Frame detections: {len(las_detections)}")
        las_tracks = las_tracker.update_tracks(las_detections, frame=las_frame)
        for las_track in las_tracks:
            if not las_track.is_confirmed():
                continue
            las_l, las_t, las_r, las_b = las_track.to_ltrb()
            las_cx, las_cy = int((las_l + las_r) / 2), int((las_t + las_b) / 2)
            las_best_det = None
            las_best_dist = float('inf')
            for las_det in las_detections:
                (las_dx1, las_dy1, las_dw, las_dh), las_score, las_label = las_det
                las_d_cx, las_d_cy = las_dx1 + las_dw / 2, las_dy1 + las_dh / 2
                las_dist = ((las_d_cx - las_cx)**2 + (las_d_cy - las_cy)**2)**0.5
                if las_dist < las_best_dist and las_dist < 50:
                    las_best_dist = las_dist
                    las_best_det = (las_dx1, las_dy1, las_dx1 + las_dw, las_dy1 + las_dh)
            if las_best_det is not None:
                las_box_to_draw = las_best_det
            else:
                las_box_to_draw = (las_l, las_t, las_r, las_b)
            las_now = time.strftime("%H:%M:%S", time.localtime())
            if las_track.track_id not in las_id_mapping:
                las_id_mapping[las_track.track_id] = f"Human{las_id_counter:05d}"
                las_id_counter += 1
            las_custom_id = las_id_mapping[las_track.track_id]
            if las_custom_id not in las_person_records:
                las_person_records[las_custom_id] = (las_now, (las_cx, las_cy))
            las_draw_fancy_box(las_frame, (las_box_to_draw[0], las_box_to_draw[1]),
                               (las_box_to_draw[2], las_box_to_draw[3]), (0, 122, 255), 2)
            cv2.putText(las_frame, f"{las_custom_id}", (int(las_box_to_draw[0]), int(las_box_to_draw[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 122, 255), 2)
        las_required_height = 60 + len(las_person_records) * 55
        if las_required_height < las_frame.shape[0]:
            las_required_height = las_frame.shape[0]
        las_panel_full = np.full((las_required_height, 300, 3), 245, dtype=np.uint8)
        cv2.putText(las_panel_full, "Rona Vision AI", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        las_y_offset_text = 60
        for las_pid, (las_first_time, (las_px, las_py)) in las_person_records.items():
            cv2.putText(las_panel_full, f"{las_pid}: ({las_px},{las_py})", (10, las_y_offset_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
            las_y_offset_text += 25
            cv2.putText(las_panel_full, f"  Time: {las_first_time}", (10, las_y_offset_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
            las_y_offset_text += 30
        las_max_scroll = las_required_height - las_frame.shape[0]
        if las_max_scroll < 0:
            las_max_scroll = 0
        if hasattr(cv2, "setTrackbarMax"):
            cv2.setTrackbarMax("Scroll", "Rona Vision AI", las_max_scroll)
        las_scroll_offset = cv2.getTrackbarPos("Scroll", "Rona Vision AI")
        las_panel = las_panel_full[las_scroll_offset:las_scroll_offset + las_frame.shape[0]]
        las_combined = np.hstack((las_frame, las_panel))
        cv2.imshow("Rona Vision AI", las_combined)
        las_key = cv2.waitKeyEx(1)
        if las_key == ord('q'):
            break
    las_cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
