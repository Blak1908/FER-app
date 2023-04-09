from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # listen for connections
    await websocket.accept()
    #count = 1
    try:
        while True:
            contents = await websocket.receive_bytes()
            arr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            #cv2.imwrite("frame%d.png" % count, frame)
            #count += 1
    except WebSocketDisconnect:
        cv2.destroyWindow("frame")
        print("Client disconnected") 