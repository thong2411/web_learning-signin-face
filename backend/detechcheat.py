# head_gesture.py
# Module phát hiện cử chỉ đầu quay trái, phải, lên, xuống

import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo MediaPipe Face Mesh

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Ngưỡng góc để xác định cử chỉ
THRESHOLD_LEFT = -15  # Quay trái
THRESHOLD_RIGHT = 15  # Quay phải
THRESHOLD_UP = -17    # Ngửa lên
THRESHOLD_DOWN = 8   # Cúi xuống

def get_head_pose(landmarks, image_shape):
    """
    Tính góc xoay của đầu (yaw, pitch, roll)
    
    Args:
        landmarks: Face landmarks từ MediaPipe
        image_shape: Kích thước ảnh (height, width)
    
    Returns:
        dict: {'yaw': góc_ngang, 'pitch': góc_dọc, 'roll': góc_nghiêng}
    """
    h, w = image_shape[:2]
    
    # Các điểm landmark quan trọng
    # 1: Mũi, 33: Mắt trái ngoài, 263: Mắt phải ngoài
    # 61: Môi trên trái, 291: Môi trên phải
    # 199: Trán giữa
    
    # Lấy tọa độ 3D của các điểm
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h, landmarks[1].z * w])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h, landmarks[152].z * w])
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h, landmarks[33].z * w])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h, landmarks[263].z * w])
    left_mouth = np.array([landmarks[61].x * w, landmarks[61].y * h, landmarks[61].z * w])
    right_mouth = np.array([landmarks[291].x * w, landmarks[291].y * h, landmarks[291].z * w])
    
    # Tính góc Yaw (quay trái/phải)
    # Dựa vào khoảng cách từ mũi đến 2 bên mắt
    left_distance = np.linalg.norm(nose_tip[:2] - left_eye[:2])
    right_distance = np.linalg.norm(nose_tip[:2] - right_eye[:2])
    yaw = (right_distance - left_distance) / (right_distance + left_distance) * 50
    
    # Tính góc Pitch (lên/xuống)
    # Dựa vào vị trí tương đối của mũi và cằm
    pitch = (nose_tip[1] - chin[1]) / h * 100
    
    # Tính góc Roll (nghiêng)
    # Dựa vào độ nghiêng của đường nối 2 mắt
    roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
    
    return {
        'yaw': float(yaw),
        'pitch': float(pitch),
        'roll': float(roll)
    }


def detect_head_gesture(frame):
    """
    Phát hiện cử chỉ đầu từ frame ảnh
    
    Args:
        frame: numpy array (BGR image)
    
    Returns:
        dict: Kết quả phát hiện
    """
    if frame is None:
        return {
            'detected': False,
            'gesture': 'Lỗi: Không có ảnh',
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'confidence': 0
        }
    
    # Chuyển sang RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(image_rgb)

    
    if not face_results.detections:
        return {
            'detected': False,
            'gesture': 'Không phát hiện khuôn mặt',
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'confidence': 0
        }
    if len(face_results.detections) >= 2:
        return {
            'detected': False,
            'gesture': 'NHIỀU NGƯỜI',
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'confidence': 1.0
        }
    mesh_results = face_mesh.process(image_rgb)

    if not mesh_results.multi_face_landmarks:
        return {
            'detected': False,
            'gesture': 'Không đủ landmark',
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'confidence': 0
        }

    face_landmarks = mesh_results.multi_face_landmarks[0].landmark    

    # Tính góc xoay đầu
    angles = get_head_pose(face_landmarks, frame.shape)
    yaw = angles['yaw']
    pitch = angles['pitch']
    roll = angles['roll']
    
    # Xác định cử chỉ
    gesture = 'Nhìn thẳng'
    confidence = 0.0
    
    # Ưu tiên theo thứ tự: Trái/Phải -> Lên/Xuống
    if yaw < THRESHOLD_LEFT:
        gesture = 'LEFT'
        confidence = min(abs(yaw) / 30, 1.0)
    elif yaw > THRESHOLD_RIGHT:
        gesture = 'RIGHT'
        confidence = min(abs(yaw) / 30, 1.0)
    elif pitch < THRESHOLD_UP:
        gesture = 'UP'
        confidence = min(abs(pitch) / 20, 1.0)
    elif pitch > THRESHOLD_DOWN:
        gesture = 'DOWN'
        print(pitch)
        confidence = min(abs(pitch) / 20, 1.0)
    else:
        gesture = 'Nhìn thẳng'
        confidence = 1.0 - min(abs(yaw) / 15 + abs(pitch) / 10, 1.0)
    
    return {
        'detected': True,
        'gesture': gesture,
        'yaw': round(yaw, 2),
        'pitch': round(pitch, 2),
        'roll': round(roll, 2),
        'confidence': round(confidence, 2)
    }


def draw_head_gesture(frame, result):
    """
    Vẽ kết quả lên frame (dùng cho testing)
    
    Args:
        frame: numpy array (BGR image)
        result: dict kết quả từ detect_head_gesture()
    
    Returns:
        frame với thông tin được vẽ lên
    """
    if not result['detected']:
        cv2.putText(frame, result['gesture'], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    # Vẽ thông tin
    gesture = result['gesture']
    yaw = result['yaw']
    pitch = result['pitch']
    confidence = result['confidence']
    
    # Màu theo cử chỉ
    color = (0, 255, 0) if gesture == 'Nhìn thẳng' else (0, 255, 255)
    
    # Hiển thị cử chỉ
    cv2.putText(frame, f"Cu chi: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Hiển thị góc
    cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame


# Test function
if __name__ == "__main__":
    print("Bắt đầu test head gesture detection...")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Phát hiện cử chỉ
        result = detect_head_gesture(frame)
        
        # Vẽ kết quả
        frame = draw_head_gesture(frame, result)
      
        cv2.imshow('Head Gesture Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()