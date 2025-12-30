import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Khởi tạo các model toàn cục
detector = MTCNN()
facenet = FaceNet()
known_faces = {}  
THRESHOLD = 0.85  


def get_face_embedding_from_frame(frame):
    """
    Trích xuất embedding từ frame (ảnh từ webcam)
    
    Args:
        frame: Frame từ webcam (numpy array)
        
    Returns:
        embedding vector hoặc None nếu không tìm thấy khuôn mặt
    """
    # Chuyển sang RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) == 0:
        return None
        
    # Lấy khuôn mặt đầu tiên
    x, y, w, h = faces[0]['box']
    
    # Đảm bảo tọa độ không âm
    x, y = max(0, x), max(0, y)
    
    face = img_rgb[y:y+h, x:x+w]
    
    # Resize về kích thước 160x160
    face = cv2.resize(face, (160, 160))
    
    # Chuẩn hóa ảnh
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    
    # Tạo embedding
    embedding = facenet.embeddings(face)[0]
    
    return embedding, (x, y, w, h)


def get_face_embedding(image_path):
    """
    Trích xuất embedding từ ảnh (dùng cho đăng ký)
    
    Args:
        image_path: Đường dẫn đến ảnh
        
    Returns:
        embedding vector hoặc None
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) == 0:
        print("Không tìm thấy khuôn mặt trong ảnh!")
        return None
        
    x, y, w, h = faces[0]['box']
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    embedding = facenet.embeddings(face)[0]
    
    return embedding


def register_face(name, image_path):
    """
    Đăng ký khuôn mặt từ ảnh
    
    Args:
        name: Tên người dùng
        image_path: Đường dẫn đến ảnh
    """
    embedding = get_face_embedding(image_path)
    
    if embedding is not None:
        known_faces[name] = embedding
        print(f"✓ Đã đăng ký khuôn mặt của {name}")
        return True
    return False


def register_face_from_webcam(name):
    """
    Đăng ký khuôn mặt trực tiếp từ webcam
    
    Args:
        name: Tên người dùng
    """
    cap = cv2.VideoCapture(0)
    print(f"Nhấn SPACE để chụp và đăng ký khuôn mặt cho {name}")
    print("Nhấn Q để hủy")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Hiển thị hướng dẫn
        cv2.putText(frame, f"Dang ky: {name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Nhan SPACE de chup", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Dang ky khuon mat', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space để chụp
            result = get_face_embedding_from_frame(frame)
            if result is not None:
                embedding, _ = result
                known_faces[name] = embedding
                print(f"✓ Đã đăng ký khuôn mặt của {name}")
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print("Không tìm thấy khuôn mặt, thử lại!")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return False


def cosine_similarity(emb1, emb2):
    """Tính độ tương đồng cosine giữa 2 embedding"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def authenticate_realtime():
    """
    Xác thực khuôn mặt TRỰC TIẾP từ webcam (real-time)
    """
    cap = cv2.VideoCapture(0)
    
    print("=== XÁC THỰC REAL-TIME ===")
    print("Nhấn Q để thoát")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Lấy embedding từ frame hiện tại
        result = get_face_embedding_from_frame(frame)
        
        if result is not None:
            embedding, (x, y, w, h) = result
            
            # So sánh với các khuôn mặt đã đăng ký
            best_match = "Unknown"
            best_similarity = -1
            
            for name, known_embedding in known_faces.items():
                similarity = cosine_similarity(embedding, known_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Xác định màu và text
            if best_similarity > THRESHOLD:
                color = (0, 255, 0)  # Xanh lá - Thành công
                label = f"{best_match} ({best_similarity:.2f})"
            else:
                color = (0, 0, 255)  # Đỏ - Không nhận diện
                label = f"Unknown ({best_similarity:.2f})"
            
            # Vẽ khung và tên
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # Không phát hiện khuôn mặt
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Hiển thị hướng dẫn
        cv2.putText(frame, "Nhan Q de thoat", (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Xac thuc khuon mat - Real-time', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    authenticate_realtime()