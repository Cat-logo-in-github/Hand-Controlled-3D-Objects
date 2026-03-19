import numpy as np

class GestureRecognizer:
    GRAB_THRESHOLD = 0.08
    PINCH_THRESHOLD = 0.035

    XY_GAIN = 12.0   # strong XY motion
    Z_GAIN = 8.0

    def __init__(self):
        self.prev_index_pos = {}
        self.prev_rot_vec = None

    def recognize_gestures(self, hands, handedness):
        results = []
        hand_centers = {}

        for i in range(len(hands)):
            if i >= len(handedness):
                continue

            hand = hands[i]
            label = handedness[i]

            index_tip = hand[8]
            thumb_tip = hand[4]
            wrist = hand[0]

            prev_index = self.prev_index_pos.get(label, index_tip)
            delta = index_tip - prev_index
            delta[1] *= -1  # flip Y (screen → world)

            gesture = "NONE"
            delta_translate = np.zeros(3)
            delta_rotate = np.zeros(3)

            # --------------------------
            # GRAB → TRANSLATION
            # --------------------------
            if np.linalg.norm(index_tip - wrist) < self.GRAB_THRESHOLD:
                gesture = "GRAB"

                if label == "Right":
                    delta_translate[0] = delta[0] * self.XY_GAIN
                    delta_translate[1] = delta[1] * self.XY_GAIN

                elif label == "Left":
                    delta_translate[2] = delta[1] * self.Z_GAIN

            # --------------------------
            # PINCH → SCALING
            # --------------------------
            if np.linalg.norm(index_tip - thumb_tip) < self.PINCH_THRESHOLD:
                gesture = "PINCH"

            # Center for rotation
            center = np.mean(hand[[0, 5, 9, 13, 17]], axis=0)
            hand_centers[label] = center

            results.append({
                "handedness": label,
                "landmarks": hand,
                "gesture": gesture,
                "delta": {
                    "translate": delta_translate,
                    "rotate": delta_rotate
                }
            })

            self.prev_index_pos[label] = index_tip.copy()

        # --------------------------
        # TWO-HAND ROTATION ONLY
        # --------------------------
        if (
            len(hand_centers) == 2 and
            all(r["gesture"] == "NONE" for r in results)
        ):
            labels = list(hand_centers.keys())
            c1, c2 = hand_centers[labels[0]], hand_centers[labels[1]]
            vec = c2 - c1

            if self.prev_rot_vec is not None:
                delta_vec = vec - self.prev_rot_vec
                rotation = np.array([
                    delta_vec[1] * 180,
                    delta_vec[0] * 180,
                    0.0
                ])

                for r in results:
                    r["gesture"] = "ROTATE"
                    r["delta"]["rotate"] = rotation

            self.prev_rot_vec = vec
        else:
            self.prev_rot_vec = None

        return results
