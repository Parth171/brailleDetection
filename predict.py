from ultralytics import YOLO

model = YOLO('./runs/classify/train3/weights/last.pt')  # load a custom model

results = model('C:/Users/User/PycharmProjects/brailleDetection/data/braille_data/val/a/a1.jpg')  # predict on an image


names = results[0].names


probs = results[0].probs.tolist()

print(names)
print(results)

# [0.24236248433589935, 0.26641324162483215, 0.24460835754871368, 0.24661587178707123]