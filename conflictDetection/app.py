import pickle
import numpy as np
import cv2
from detectConflict import DetectionAnalysis as pe

def predict_for_each_person(distances, model, threshold=0.67):
    predictions = []
    for distance in distances:
        if distance:
            dis = np.array(distance).reshape(1, -1)
            if dis.shape[1] == 350:
                probas = model.predict_proba(dis)
                max_proba = np.max(probas)
                if max_proba >= threshold:
                    prediction = model.predict(dis)
                    predictions.append((int(prediction[0]), max_proba))
                else:
                    predictions.append(max_proba)
    return predictions

def process_frame(frame, estimator, model):
    process = estimator.process_image(frame)
    distances = estimator.detect_contact(frame)

    if isinstance(distances, list) and all(isinstance(d, list) for d in distances):
        predictions = predict_for_each_person(distances, model)
    else:
        predictions = ["Invalid distance data"]

    return predictions

def main():
    # Load the pose estimator and model
    estimator = pe(model_path='yolov8n-pose.pt')
    model_dict = pickle.load(open('models/model99999.p', 'rb'))
    model = model_dict['model']
    
    # Open a video capture from the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        predictions = process_frame(frame, estimator, model)

        # Check if any person is fighting
        fighting = False
        not_fighting = False

        for idx, prediction in enumerate(predictions):
            if prediction is not None:
                # Check if prediction is a tuple
                    if prediction[0] > 0.75:
                        fighting = True
                    else:
                        not_fighting = True
                # Check if prediction is a number
                    if prediction > 0.75:
                        fighting = True
                    else:
                        not_fighting = True

        # Display overall status
        if fighting:
            cv2.putText(frame, 'Fighting', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif not_fighting:
            cv2.putText(frame, 'Not Fighting', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'No Action', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Pose Estimation', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


Not = False
str = str(input('enter word:'))
for i in len(str):
    if str[i:i+3]=='not':
        Not = True
if Not == False:
    print('not ', str)
else:
    print(str)