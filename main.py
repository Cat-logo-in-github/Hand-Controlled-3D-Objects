import time
import numpy as np
import cv2

from ar_powered_design.objects.cuboid import Cuboid
from ar_powered_design.physics import PhysicsEngine
from ar_powered_design.gestures import GestureRecognizer
from ar_powered_design.hand_tracker import HandTracker
from ar_powered_design.renderer import Renderer
from ar_powered_design.ui import draw_hand_landmarks, overlay_text

# --------------------------
# Initialize components
# --------------------------
renderer = Renderer(width=800, height=600, title="AR Powered Designer")
physics = PhysicsEngine()
gestures = GestureRecognizer()
tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)

cube = Cuboid(width=1.0, height=1.0, depth=1.0)
renderer.add_object(cube)

prev_two_hand_distance = None

# --------------------------
# Main loop
# --------------------------
cap = cv2.VideoCapture(0)

try:
    while not renderer.should_close():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # --------------------------
        # Detect hands
        # --------------------------
        hands_data = tracker.detect_hands(frame)
        hands = hands_data["hands"]
        handedness = hands_data["handedness"]

        # --------------------------
        # Recognize gestures (PER HAND)
        # --------------------------
        gesture_results = gestures.recognize_gestures(hands, handedness)

        # --------------------------
        # Map gestures to object actions
        # --------------------------
        delta_translation = np.zeros(3)
        delta_rotation = np.zeros(3)
        delta_scale = np.ones(3)

        if len(gesture_results) == 2:
            h1, h2 = gesture_results

            # Two-hand scaling (pinch)
            if h1["gesture"] == "PINCH" and h2["gesture"] == "PINCH":
                center1 = (h1["landmarks"][4] + h1["landmarks"][8]) / 2
                center2 = (h2["landmarks"][4] + h2["landmarks"][8]) / 2
                dist = np.linalg.norm(center1 - center2)

                if prev_two_hand_distance:
                    scale = dist / prev_two_hand_distance
                    delta_scale = np.array([scale, scale, scale])
                prev_two_hand_distance = dist

            # Translation
            for h in gesture_results:
                if h["gesture"] == "GRAB":
                    delta_translation += h["delta"]["translate"]

            delta_translation *= 0.5

            # Rotation
            for h in gesture_results:
                if h["gesture"] == "ROTATE":
                    delta_rotation += h["delta"]["rotate"]

        elif len(gesture_results) == 1:
            h = gesture_results[0]
            if h["gesture"] == "GRAB":
                delta_translation = h["delta"]["translate"]
            elif h["gesture"] == "ROTATE":
                delta_rotation = h["delta"]["rotate"]

        # --------------------------
        # Apply physics
        # --------------------------
        physics.apply(cube, {
            "translate": delta_translation,
            "rotate": delta_rotation,
            "scale": delta_scale
        })

        # --------------------------
        # Draw UI
        # --------------------------
        overlay_text(frame, f"Objects: {len(renderer.objects)}", (10, 30))

        # Draw RAW HANDS (not gestures)
        draw_hand_landmarks(
            frame,
            [
                {"landmarks": hands[i], "handedness": handedness[i]}
                for i in range(len(hands))
            ]
        )

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        renderer.render_scene()

finally:
    cap.release()
    cv2.destroyAllWindows()
    renderer.terminate()
