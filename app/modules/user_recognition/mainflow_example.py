from deepface import DeepFace
import cv2
import os
import base64
from time import time

num_frames = 5  # Number of frames to capture
output_folder = "frames"  # Folder to save the frames

cap = cv2.VideoCapture(0)  # Use 0 for webcam, or specify the video file path
frames = []

while len(frames) < num_frames:
    # Read a frame from the video source
    ret, frame = cap.read()

    # Generate a unique filename for the frame
    filename = f"frame{len(frames)+1}.jpg"
    filepath = os.path.join(output_folder, filename)

    # Save the frame as an image
    cv2.imwrite(filepath, frame)

    # Append the file path to the list of frames
    frames.append(filepath)

    # Display the frame
    cv2.imshow("Capture Frames", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

results = []

start = time()
for frame_path in frames:
    result = DeepFace.analyze(frame_path, actions=['emotion', 'age', 'race', 'gender'], enforce_detection=False)
    results.append(result)
stop = time()
print("----Time process----:", stop-start)
combined_result = {
    'emotion': {},
    'age': 0,
    'gender': {},
    'race': {}
}

for result in results:
    last_result = result[-1]
    for emotion, probability in last_result['emotion'].items():
        if emotion not in combined_result['emotion'] or probability > combined_result['emotion'][emotion]:
            combined_result['emotion'][emotion] = probability
    combined_result['age'] += last_result['age']
    for gender, probability in last_result['gender'].items():
        if gender not in combined_result['gender'] or probability > combined_result['gender'][gender]:
            combined_result['gender'][gender] = probability
    for race, probability in last_result['race'].items():
        if race not in combined_result['race'] or probability > combined_result['race'][race]:
            combined_result['race'][race] = probability

emotion = max(combined_result['emotion'], key=combined_result['emotion'].get)
age = combined_result['age'] / len(results)
gender = max(combined_result['gender'], key=combined_result['gender'].get)
race = max(combined_result['race'], key=combined_result['race'].get)

print("Combined Result:")
print("Emotion:", emotion)
print("Age:", int(age))
print("Gender:", gender)
print("Race:", race)

# Save the combined result to a text file
output_file = "combined_result.txt"
with open(output_file, 'w') as file:
    file.write("Combined Result:\n")
    file.write("Emotion: " + emotion + "\n")
    file.write("Average Age: " + str(age) + "\n")
    file.write("Gender: " + gender + "\n")
    file.write("Race: " + race + "\n")
    file.write("\n")

print("Combined result saved to", output_file)