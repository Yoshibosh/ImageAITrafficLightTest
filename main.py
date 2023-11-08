import os
import pathlib

from imageai.Classification.Custom import CustomImageClassification, ClassificationModelTrainer
from imageai.Detection.Custom import DetectionModelTrainer, CustomObjectDetection


def main():
    pass
    # classifyObject('mnist_cnn.pt', 'Images/RFpasport_cifra_6.png')


def classifyObject(model_path, img_path):
    prediction = CustomImageClassification()
    prediction.setModelPath(model_path)
    prediction.loadModel()

    predictions, probabilities = prediction.classifyImage(img_path, result_count=2)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, ": ", eachProbability)

def trainObjectClassification():
    lastModelPath = "./yolo3.pt"

    trainer = ClassificationModelTrainer()
    trainer.setModelTypeAsMobileNetV2()
    trainer.setDataDirectory(data_directory='mnist_digits')
    trainer.trainModel(num_experiments=14)


def detectObject():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath('traffic_lights/models/yolov3_traffic_lights_last.pt')
    detector.setJsonPath('traffic_lights/json/traffic_lights_yolov3_detection_config.json')
    detector.loadModel()
    detections = detector.detectObjectsFromImage(
        input_image="Images/img_105.jpg",
        output_image_path="Images/img_105_detect.jpg"
    )
    for detection in detections:
        print(detection['name'], " : ", detection['percentage_probability'],
              " : ", detection['box_points'])


def trainObjectDetection():
    lastModelPath = "./yolo3.pt"

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory='traffic_lights')
    trainer.setTrainConfig(object_names_array=["tl"],
                           num_experiments=200,
                           train_from_pretrained_model=lastModelPath)
    trainer.trainModel()


def changeFileNames():
    path = str(pathlib.Path().resolve()) + '/traffic_lights/train/images/'
    directory = os.fsencode(path)
    count = 1

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filename, extension = os.path.splitext(filename)
        if extension.endswith(".jpg") or extension.endswith(".jpeg") or extension.endswith(".png"):
            # print(path + 'img_' + str(count) + extension)
            os.rename(path + filename + extension, path + 'img_' + str(count) + extension)
            count += 1
            continue
        else:
            continue


if __name__ == '__main__':
    main()
