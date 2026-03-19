import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

HAND_COLORS = {
    "Right": (0,255,0),
    "Left": (0,0,255)
}

def draw_hand_landmarks(frame, hands_data):
    h, w, _ = frame.shape

    for hand in hands_data:
        landmarks = hand["landmarks"]
        label = hand.get("handedness", "Unknown")
        color = HAND_COLORS.get(label, (255,255,255))

        pts = [(int(lm[0]*w), int(lm[1]*h)) for lm in landmarks]

        for a,b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)

        for i,(x,y) in enumerate(pts):
            r = 5 if i in [4,8,12,16,20] else 3
            cv2.circle(frame, (x,y), r, color, -1)

def overlay_text(frame, text, pos=(10,30)):
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2)
