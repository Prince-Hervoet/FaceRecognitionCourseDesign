import numpy
import os
import config
import cv2
import util

kTrainDataPath = config.kTrainDataPath
kTestSetPath = config.kTestSetPath
kFaceCascade = config.kFaceCascade

"""
matLikeImgInfo 是 [图片名称, MatLike类型的图片]
"""


# 获得idName映射字典和识别器
def getRecognizer():
    idToName = numpy.load(
        os.path.join(kTrainDataPath, config.kIdToNameFileName) + ".npy",
        allow_pickle=True,
    ).item()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(kTrainDataPath, config.kTrainDataFileName))
    return idToName, recognizer


# 从文件数据中识别
def recognizeFromFiles():
    matLikeImgInfos = []
    fileNames = os.listdir(kTestSetPath)
    idToName, recognizer = getRecognizer()
    for fileName in fileNames:
        matLikeImg = cv2.imread(os.path.join(kTestSetPath, fileName))
        matLikeImgInfos.append([fileName, matLikeImg])
        ans = recognizeAnalyse(matLikeImg, idToName, recognizer)
        for info in ans:
            print("根据 " + fileName + " 识别到: " + info["name"])


# 从内存数据中识别
def recognizeFromData(matLikeImg, idToName, recognizer):
    return recognizeAnalyse(matLikeImg, idToName, recognizer)


# 识别测试数据集中的人脸
def recognizeAnalyse(matLikeImg, idToName, recognizer):
    ans = []
    faceBox = util.getFace(matLikeImg)
    if len(faceBox) == 0:
        return ans
    for index in range(len(faceBox)):
        peopleName = config.kUnknownStr
        (x, y, w, h) = faceBox[index]
        # goal越大越不相似
        gray = util.getGray(matLikeImg)
        id, goal = recognizer.predict(gray[y : y + h, x : x + w])
        if goal <= 70:
            peopleName = idToName.get(id)
        ans.append({"name": peopleName, "oneFaceBox": faceBox[index]})
    return ans
