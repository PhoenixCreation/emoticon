# DeepFace library for face and emotion detection
from deepface import DeepFace
# Opencv library for image manupulation
import cv2
# Classic time library for fps counting, shall be removed in production build
import time

# Get the camera
# vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture("./test_video.mp4")
# Use cv2.VideoCapture("./test_video.mp4") for testing video
# If you have multiple camera, you can adjust the number as per your camera, 0 is for default camera

# Pre trained models for gender prediction
# We are not using deepface for gender prediction because it needs to download 500MB model which is not practical
genderProto = "model/gender_deploy.prototxt"
genderModel = "model/gender_net.caffemodel"


# Pre defined values for gender model, changing this may break the model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['male', 'female']
genderNet = cv2.dnn.readNet(genderModel, genderProto)
padding = 20


# Dict for holding the images related to the emotion
# Key names are given as per the return value of prediction function to make life easier
# It has nested dictionary for gender based emoticons
emoji_img = {
    "angry": cv2.imread("./emojis/angry.png"),
    "disgust": cv2.imread("./emojis/disgust.png"),
    "fear": cv2.imread("./emojis/fear.png"),
    "happy": cv2.imread("./emojis/happy.png"),
    "neutral": cv2.imread("./emojis/neutral.png"),
    "sad": cv2.imread("./emojis/sad.png"),
    "surprise": cv2.imread("./emojis/surprise.png"),
    "male": {
        "angry": cv2.imread("./emojis/male/male_angry.png"),
        "disgust": cv2.imread("./emojis/male/male_disgust.png"),
        "fear": cv2.imread("./emojis/male/male_fear.png"),
        "happy": cv2.imread("./emojis/male/male_happy.png"),
        "neutral": cv2.imread("./emojis/male/male_neutral.png"),
        "sad": cv2.imread("./emojis/male/male_sad.png"),
        "surprise": cv2.imread("./emojis/male/male_surprise.png")
    },
    "female": {
        "angry": cv2.imread("./emojis/female/female_angry.png"),
        "disgust": cv2.imread("./emojis/female/female_disgust.png"),
        "fear": cv2.imread("./emojis/female/female_fear.png"),
        "happy": cv2.imread("./emojis/female/female_happy.png"),
        "neutral": cv2.imread("./emojis/female/female_neutral.png"),
        "sad": cv2.imread("./emojis/female/female_sad.png"),
        "surprise": cv2.imread("./emojis/female/female_surprise.png")
    },
}

# list which keeps the track of the last emotions,number of frames is defined below
last_emotions = []

# Infinity while loop
while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Time keeper for fps counter
    start = time.time()
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

    # variable that stores the final information
    result = {"gender": "male", "emotion": "neutral"}

    # If x and y both are 0 that means no face was detected
    if x != 0 and y != 0:
        # First predict the gender of the face
        faceBox = [x, y, x+w, y+h]
        face = frame[max(0, faceBox[1]-padding):
                     min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        result["gender"] = gender

        cv2.putText(frame, gender, (x, y+h+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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

    result["emotion"] = emotion['dominant_emotion']

    # This is FPS counter, actually it is NOT real fps as it only takes calculation time in count
    cv2.putText(frame, f'{round(1 / (time.time() - start))}', (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Display the video
    cv2.imshow('original', frame)

    # Get the copy of predicted gender and emotion
    emoticon = emoji_img[result["gender"].lower()][result["emotion"]].copy()

    # This is for emoji window, put the lable of emotion in bottom-left corner
    cv2.putText(emoticon, result["emotion"],
                (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(emoticon, result["gender"],
                (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('emoticon', emoticon)

    # Press 'q' to quit the app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clanup after closing
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
