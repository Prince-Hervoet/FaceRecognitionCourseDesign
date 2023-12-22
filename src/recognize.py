import numpy
import os
import config
import cv2

kTrainDataPath = config.kTrainDataPath
kTestSetPath = config.kTestSetPath
kFaceCascade = config.kFaceCascade


# 识别测试数据集中的人脸
def recognize():
    idToName = numpy.load(
        os.path.join(kTrainDataPath, config.kIdToNameFileName) + ".npy",
        allow_pickle=True,
    ).item()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(kTrainDataPath, config.kTrainDataFileName))
    fileNames = os.listdir(kTestSetPath)
    for fileName in fileNames:
        matLikeImg = cv2.imread(os.path.join(kTestSetPath, fileName))
        gray = cv2.cvtColor(matLikeImg, cv2.COLOR_BGR2GRAY)
        faceBox = kFaceCascade.detectMultiScale(gray)  # 获得脸部位置信息
        for x, y, w, h in faceBox:
            # goal越大越不相似
            id, goal = recognizer.predict(gray[y : y + h, x : x + w])
            if goal <= 80:
                print("根据 " + fileName + " 检测到: " + idToName.get(id))
            else:
                print("根据 " + fileName + " 检测到: " + "unknown")
