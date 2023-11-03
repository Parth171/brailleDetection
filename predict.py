from ultralytics import YOLO

import cv2

model = YOLO('./runs/classify/train7/weights/best.pt')  # load a custom model


vid = cv2.VideoCapture(0) # making the video variable and defining the webcam



while True:

    _, frame = vid.read() # reading the video variable and storing it in frame variable
    cv2.imshow("Camera", frame)

    results = model(frame)  # predict on an image

    names = results[0].names

    probs = results[0].probs.tolist()

    probability = 0
    index = 0

    #print(names) aaa
    #print(probs)sss

    for i in range(len(probs)):
        if probs[i] > probability:
            probability = probs[i]

    for i in range(len(probs)):
        if probs[i] == probability:
            index = i

    letter = names[index]

    if probability < 0.6:
        print("no letter detected")

    else:

        print(f"The letter shown is: {letter}")


    if cv2.waitKey(1) == ord("e"):
        break

vid.release()  # releases the resource (webcam in this case)
cv2.destroyAllWindows()

print("Commit check")

# it can detect 'b' fine







