# server.py
# Backend FastAPI cho hệ thống nhận diện khuôn mặt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import numpy as np
import pickle

# Import từ file main.py
from main import (
    register_face,
    known_faces,
    cosine_similarity,
    get_face_embedding_from_frame,
    get_face_embedding,
    THRESHOLD
)

app = FastAPI(title="Face Recognition API")

# Cấu hình CORS (cho phép frontend gọi API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
DATA_DIR = "face_data"
FACES_FILE = os.path.join(DATA_DIR, "faces.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# === HÀM LƯU/TẢI DỮ LIỆU ===

def save_faces():
    """Lưu known_faces vào file"""
    try:
        with open(FACES_FILE, 'wb') as f:
            pickle.dump(known_faces, f)
        print(f"Đã lưu {len(known_faces)} khuôn mặt vào {FACES_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu: {e}")


def load_faces():
    """Tải known_faces từ file khi khởi động server"""
    global known_faces
    try:
        if os.path.exists(FACES_FILE):
            with open(FACES_FILE, 'rb') as f:
                loaded_faces = pickle.load(f)
                known_faces.clear()
                known_faces.update(loaded_faces)
            print(f"Đã tải {len(known_faces)} khuôn mặt từ {FACES_FILE}")
        else:
            print("Chưa có dữ liệu khuôn mặt nào được lưu")
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")


# Tải dữ liệu khi server khởi động
@app.on_event("startup")
async def startup_event():
    load_faces()
    


# Lưu dữ liệu khi server tắt
@app.on_event("shutdown")
async def shutdown_event():
    save_faces()
    



@app.get("/")
def root():
    """Kiểm tra API đang chạy"""
    return {
        "status": "running",
        "message": "Face Recognition API is ready",
        "registered_users": len(known_faces)
    }


@app.post("/register")
async def register_user(name: str = Form(...), image: UploadFile = File(...)):
    """
    Đăng ký khuôn mặt từ ảnh upload
    
    Args:
        name: Tên người dùng
        image: File ảnh upload
    """
    try:
        # Tạo tên file không dấu, dùng timestamp để tránh trùng
        import time
        timestamp = str(int(time.time()))
        file_path = os.path.join(UPLOAD_DIR, f"user_{timestamp}.jpg")
        
        # Lưu file tạm
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Đăng ký khuôn mặt
        success = register_face(name, file_path)
        
        # Xóa file tạm sau khi xử lý
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if success:
            # Lưu dữ liệu ngay sau khi đăng ký thành công
            save_faces()
            
            return {
                "status": "success",
                "message": f"Đã đăng ký khuôn mặt của {name}",
                "name": name,
                "total_users": len(known_faces)
            }
        else:
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh")
            
    except Exception as e:
        # Xóa file nếu có lỗi
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


@app.post("/authenticate")
async def authenticate(image: UploadFile = File(...)):
    """
    Xác thực khuôn mặt từ ảnh upload
    
    Args:
        image: File ảnh cần xác thực
    """
    try:
        # Đọc ảnh từ request
        image_bytes = await image.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Không thể đọc ảnh")
        
        # Lấy embedding
        result = get_face_embedding_from_frame(frame)
        
        if result is None:
            return {
                "status": "no_face",
                "message": "Không phát hiện khuôn mặt trong ảnh"
            }
        
        embedding, (x, y, w, h) = result
        
        # So sánh với database
        best_match = "Unknown"
        best_similarity = -1
        
        for name, known_embedding in known_faces.items():
            similarity = cosine_similarity(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # Kết quả
        if best_similarity > THRESHOLD:
            return {
                "status": "success",
                "result": "authenticated",
                "name": best_match,
                "similarity": float(best_similarity),
                "confidence": f"{best_similarity * 100:.1f}%"
            }
        else:
            return {
                "status": "failed",
                "result": "unknown",
                "message": "Không nhận diện được",
                "similarity": float(best_similarity),
                "confidence": f"{best_similarity * 100:.1f}%"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


@app.get("/users")
def list_users():
    """Xem danh sách người dùng đã đăng ký"""
    return {
        "total": len(known_faces),
        "users": list(known_faces.keys())
    }


@app.delete("/users/{name}")
def delete_user(name: str):
    """Xóa người dùng khỏi hệ thống"""
    if name in known_faces:
        del known_faces[name]
        # Lưu lại sau khi xóa
        save_faces()
        return {
            "status": "success",
            "message": f"Đã xóa {name}",
            "total_users": len(known_faces)
        }
    else:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy {name}")


@app.post("/backup")
def backup_data():
    """Backup dữ liệu thủ công"""
    save_faces()
    return {
        "status": "success",
        "message": "Đã backup dữ liệu",
        "total_users": len(known_faces),
        "file": FACES_FILE
    }


@app.post("/restore")
def restore_data():
    """Khôi phục dữ liệu từ file"""
    load_faces()
    return {
        "status": "success",
        "message": "Đã khôi phục dữ liệu",
        "total_users": len(known_faces)
    }


@app.get("/health")
def health_check():
    """Kiểm tra trạng thái server"""
    return {
        "status": "healthy",
        "registered_users": len(known_faces),
        "threshold": THRESHOLD
    }
# uvicorn server:app --reload --host 0.0.0.0 --port 8000