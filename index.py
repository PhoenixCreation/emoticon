# DeepFace library for face and emotion detection
from deepface import DeepFace
# Opencv library for image manupulation
import cv2

# Get the camera
vid = cv2.VideoCapture(0)
# vid = cv2.VideoCapture("./test_video.mp4")
# Use cv2.VideoCapture("./test_video.mp4") for testing video
# If you have multiple camera, you can adjust the number as per your camera, 0 is for default camera


# Dict for holding the images related to the emotion
# Key names are given as per the return value of prediction function to make life easier
emoji_img = {
    "angry": cv2.imread("./emojis/angry.png"),
    "disgust": cv2.imread("./emojis/disgust.png"),
    "fear": cv2.imread("./emojis/fear.png"),
    "happy": cv2.imread("./emojis/happy.png"),
    "neutral": cv2.imread("./emojis/neutral.png"),
    "sad": cv2.imread("./emojis/sad.png"),
    "surprise": cv2.imread("./emojis/surprise.png")
}

# list which keeps the track of the last emotions,number of frames is defined below
last_emotions = []

# Infinity while loop
while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()

    # resizing the video for better performance, can be commented if your system has ufo rating
    frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_NEAREST)

    # Predict the emotion through the deepface library
    emotion = DeepFace.analyze(
        frame, actions=['emotion'], enforce_detection=False)

    # # Extracting coordinates of the face
    x = emotion['region']['x']
    y = emotion['region']['y']
    w = emotion['region']['w']
    h = emotion['region']['h']

    cv2.putText(frame, emotion['dominant_emotion'], (x, y+h+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # # If x and y both are 0 that means no face was detected
    if x != 0 and y != 0:
        # We only keep track of last 10 emotions, this figure needs to be calibered in future
        if len(last_emotions) >= 10:
            # Pop out the first element, This will work as a QUEUE data structure, first in first out
            last_emotions.pop(0)
        # append the current emotion at the end
        last_emotions.append(emotion['dominant_emotion'])

        # Count the total number of times one expression is shown, store them in list of touples
        counter = [
            ("angry", last_emotions.count("angry")),
            ("disgust", last_emotions.count("disgust")),
            ("fear", last_emotions.count("fear")),
            ("happy", last_emotions.count("happy")),
            ("neutral", last_emotions.count("neutral")),
            ("sad", last_emotions.count("sad")),
            ("surprise", last_emotions.count("surprise"))
        ]
        # Sort the counter list based on second element of touple in decending order
        # This will give the average emation of last 10 frames as a first element
        counter.sort(key=lambda e: e[1], reverse=True)

        # Set the average emotion for further usage
        emotion['dominant_emotion'] = counter[0][0]
        # draw a rectangle arounf the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        # put the emotion tag above the rectangle
        cv2.putText(frame, emotion['dominant_emotion'], (x, y-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # If x and y both are 0 then set the emotion to "neutral"
    else:
        emotion['dominant_emotion'] = "neutral"

    # Display the video
    cv2.imshow('original', frame)

    # This is for emoji window, put the lable of emotion in bottom-left corner
    cv2.putText(emoji_img[emotion['dominant_emotion']], emotion['dominant_emotion'],
                (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('emoticon', emoji_img[emotion['dominant_emotion']])

    # Press 'q' to quit the app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clanup after closing
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
