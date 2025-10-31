import cv2
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

app = FastAPI()
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
templates = Jinja2Templates(directory="templates")

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#def gen_frames():
#    while True:
#        success, frame = camera.read()
#        if not success:
#            break
#        else:
#            ret, buffer = cv2.imencode('.jpg', frame)
#            frame = buffer.tobytes()
#            yield (b'--frame\r\n'
#                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#        time.sleep(0.03)

#def gen_frames():
#    while True:
#        success, frame = camera.read()
#        if not success:
#            break
#
#        # Convert to grayscale (better for detection)
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#        # Detect faces
#        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#        # Draw rectangles around detected faces
#        for (x, y, w, h) in faces:
#            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#        # Encode frame to JPEG
#        ret, buffer = cv2.imencode('.jpg', frame)
#        frame = buffer.tobytes()
#
#        yield (b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#        time.sleep(0.03)

def gen_frames():
    prev_time = 0  # for FPS calculation
    latency_log = []  # optional: track average latency

    while True:
        start_time = time.time()  # frame start time

        success, frame = camera.read()
        if not success:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- Calculate FPS ---
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        # --- Calculate latency ---
        latency = (current_time - start_time) * 1000  # in milliseconds
        latency_log.append(latency)
        if len(latency_log) > 30:  # keep average of last 30 frames
            latency_log.pop(0)
        avg_latency = sum(latency_log) / len(latency_log)

        # --- Draw text on frame ---
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Latency: {avg_latency:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    uvicorn.run(app, host='192.168.118.223', port=57577, reload=True)