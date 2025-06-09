import cv2
import numpy as np
from ultralytics import YOLO

class DetectionAnalysis:
    def __init__(self, model_path='yolov8n-pose.pt',draw = False):
        self.model = YOLO(model_path)
        self.PpKeys = []
        self.draw = draw
        self.max_pairs = 30  # number of pairs

    def draw_keypoints(self, frame, people_keypoints):
        self.PpKeys = []
        for keypoints in people_keypoints:
            keys = []
            for kp in keypoints:
                if len(kp) >= 2:
                    x, y = kp[:2]
                    conf = kp[2] if len(kp) > 2 else 1.0
                    if conf > 0.5:  # Confidence threshold
                        if self.draw:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        keys.append([int(x), int(y)])
                    else:
                        keys.append([0, 0])  # Use zero for low-confidence keypoints
                else:
                    keys.append([0, 0])  #Use zero for incomplete keypoints
            self.PpKeys.append(keys)

    def handle_hidden_keypoints(self, keypoints):
        """Estimate missing keypoints by referencing neighboring keypoints."""
        keypoints = keypoints[:]

        # Handle symmetrical keypoints
        symmetric_pairs = {
            5: 6,  # Shoulders
            6: 5,
            9: 10, # Hips
            10: 9,
            7: 8,  # Elbows
            8: 7,
            11: 12, # Knees
            12: 11,
            13: 14, # Ankles
            14: 13,
            15: 16, # Wrists
            16: 15
        }

        # Estimate symmetrical positions
        for left, right in symmetric_pairs.items():
            if keypoints[left] == [0, 0] and keypoints[right] != [0, 0]:
                keypoints[left] = keypoints[right]
            elif keypoints[right] == [0, 0] and keypoints[left] != [0, 0]:
                keypoints[right] = keypoints[left]

        # Estimate head position from shoulders
        if keypoints[0] == [0, 0]:
            if keypoints[3] != [0, 0] and keypoints[4] != [0, 0]:
                keypoints[0] = [
                    (keypoints[3][0] + keypoints[4][0]) / 2,
                    (keypoints[3][1] + keypoints[4][1]) / 2
                ]
            elif keypoints[3] != [0, 0]:
                keypoints[0] = keypoints[3]
            elif keypoints[4] != [0, 0]:
                keypoints[0] = keypoints[4]
        
        # Handle mid-points like chest if they are defined, as an average of left and right sides
        if keypoints[1] == [0, 0]:
            if keypoints[5] != [0, 0] and keypoints[6] != [0, 0]:  # Shoulders
                keypoints[1] = [
                    (keypoints[5][0] + keypoints[6][0]) / 2,
                    (keypoints[5][1] + keypoints[6][1]) / 2
                ]
            elif keypoints[5] != [0, 0]:
                keypoints[1] = keypoints[5]
            elif keypoints[6] != [0, 0]:
                keypoints[1] = keypoints[6]

        return keypoints

    def detect_contact(self, frame):
        results = self.PpKeys
        self.dis = []
        pairs = [
                (9, 0), (9, 1), (9, 2), (9, 3),
                (9, 4), (9, 5), (9, 6), (9, 7),
                (9, 8), (9, 9), (9, 10), (7, 0), (7, 1),
                (7, 2), (7, 3), (7, 4), (8, 0),
                (8, 1), (8, 2), (8, 3), (8, 4),
                (10, 1), (10, 2), (10, 3), (10, 4),
                (10, 5), (10, 6), (10, 7), (10, 8),
                (10, 9), (10, 10), (7, 5), (7, 6),
                (7, 7), (7, 8), (7, 9), (7, 10),
                (8, 5), (8, 6), (8, 7), (8, 8),
                (8, 9), (8, 10), (5, 6), (6, 5),
                (5, 1), (5, 2), (5, 3), (5, 4),
                (5, 5), (5, 0), (6, 0),
                (6, 2), (6, 3), (6, 4), (6, 6),
                (0, 9), (1, 9), (2, 9), (3, 9),
                (4, 9), (5, 9), (6, 9), (7, 9),
                (8, 9), (10, 9), (0, 7), (1, 7),
                (2, 7), (3, 7), (4, 7), (0, 8),
                (1, 8), (2, 8), (3, 8), (4, 8),
                (1, 10), (2, 10), (3, 10), (4, 10),
                (5, 10), (6, 10), (7, 10), (8, 10),
                (5, 7), (6, 7), 
                (9, 7), (10, 7), (5, 8), (6, 8),
                (9, 8), (10, 8), (0, 5), (1, 5),
                (2, 5), (3, 5), (4, 5), (0, 6),
                (1, 6), (2, 6), (3, 6), (4, 6),(0, 0),
                (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 0), (1, 1), (1, 2), (1, 3),
                (1, 4), (2, 0), (2, 1), (2, 2),
                (2, 3), (2, 4), (3, 0), (3, 1),
                (3, 2), (3, 3), (3, 4), (4, 0),
                (4, 1), (4, 2), (4, 3), (4, 4),
                (11, 11), (11, 12), (12, 11), (12, 12),
                (11, 0), (11, 1), (11, 2), (11, 3),
                (11, 4), (11, 5), (11, 6), (11, 7),
                (11, 8), (11, 9), (11, 10), (12, 0),
                (12, 1), (12, 2), (12, 3), (12, 4),
                (12, 5), (12, 6), (12, 7), (12, 8),
                (12, 9), (12, 10), (0, 11), (1, 11),
                (2, 11), (3, 11), (4, 11), (5, 11),
                (6, 11), (7, 11), (8, 11), (9, 11),
                (10, 11), (0, 12), (1, 12), (2, 12),
                (3, 12), (4, 12), (5, 12), (6, 12),
                (7, 12), (8, 12), (9, 12), (10, 12),
            ]


        for i in range(len(results)):
            self.ConDis = []
            for j in range(i + 1, len(results)):
                distances = [None] * len(pairs)
                for k, (idx1, idx2) in enumerate(pairs):
                    results[i] = self.handle_hidden_keypoints(results[i])
                    results[j] = self.handle_hidden_keypoints(results[j])

                    if (idx1 < len(results[j]) and idx2 < len(results[i]) and
                            results[j][idx1] != [0, 0] and results[i][idx2] != [0, 0]):

                        p1 = tuple(map(int, results[j][idx1]))
                        p2 = tuple(map(int, results[i][idx2]))

                        x_diff = abs(p1[0] - p2[0])
                        y_diff = abs(p1[1] - p2[1])
                        distances[k] = [x_diff, y_diff]
                        
                        if self.draw:
                            cv2.line(frame, p1, p2, (0, 255, 255), 1)  

                for k in range(len(distances)):
                    if distances[k] is None:
                        detected_distances = [d for d in distances if d is not None]
                        if detected_distances:
                            avg_distance = [
                                sum(d[0] for d in detected_distances) / len(detected_distances),
                                sum(d[1] for d in detected_distances) / len(detected_distances)
                            ]
                            distances[k] = avg_distance
                        else:
                            distances[k] = [0, 0]
                # Normalize distances
                distances = np.array(distances)
                max_vals = np.max(distances, axis=0, where=(distances != [0, 0]), initial=1)  # Use initial=1 to avoid division by zero
                normalized = distances / max_vals

                min_normalized = np.min(normalized, axis=0)
                adjusted_normalized = normalized - min_normalized

                scaled_normalized = (adjusted_normalized - np.min(adjusted_normalized)) / np.ptp(adjusted_normalized)

                clipped_normalized = np.clip(scaled_normalized, 0, 1)

                self.ConDis.append(clipped_normalized.tolist())

            self.dis.append(self.ConDis)

        return self.dis

    def process_image(self, frame):
        img = frame[..., ::-1]  # Convert BGR to RGB

        results = self.model(img)
        people_keypoints = []
        for result in results:
            if result.keypoints is not None:
                for keypoints in result.keypoints.xy:
                    person_keypoints = []
                    for x, y in keypoints:
                        person_keypoints.append([x.item(), y.item()])
                    people_keypoints.append(person_keypoints)

        self.draw_keypoints(frame, people_keypoints)
        

        return people_keypoints
