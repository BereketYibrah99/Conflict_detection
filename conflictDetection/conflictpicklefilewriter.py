from detectConflict import DetectionAnalysis as DA
import cv2 as cv
import os
import pickle
dir_path = 'assets/TrainTestDatas/'

def reduce_one_dimension(two_d_list):
    """Flatten a 2D list into a 1D list."""
    return [item for sublist in two_d_list for item in sublist]

estimator = DA(model_path='models/yolov8n-pose.pt',draw=True)
data = []
label = []

for subdir in os.listdir(dir_path):
    subdir_path = os.path.join(dir_path, subdir)
    
    if os.path.isdir(subdir_path):
        for image_name in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, image_name)
            
            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame = cv.imread(image_path)
                
                if frame is None:
                    print(f"Failed to load image {image_path}")
                    continue
                
                processed_image = estimator.process_image(frame)
                distance = estimator.detect_contact(frame)
                
                if isinstance(distance, (list, tuple)) and len(distance) > 0:
                    if isinstance(distance[0], (list, tuple)) and len(distance[0]) > 0:
                        print(f"{len(data)}")
                        data.append(reduce_one_dimension(distance[0][0]))
                        label.append(subdir)
print(len(label))
with open('assets/datasets/data63.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': label}, f)
print("Data processing complete. Pickle file created.")
cv.destroyAllWindows()


