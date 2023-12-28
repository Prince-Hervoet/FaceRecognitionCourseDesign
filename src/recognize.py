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
    ans = recognizeAnalyse(matLikeImgInfos, idToName, recognizer)
    for key in ans:
        print(key + " -- " + str(ans[key]))


# 从内存数据中识别
def recognizeFromData(matLikeImg, idToName, recognizer):
    imgName = util.getUUIDStr()
    return recognizeAnalyse([[imgName, matLikeImg]], idToName, recognizer)


# 识别测试数据集中的人脸
def recognizeAnalyse(matLikeImgInfos, idToName, recognizer):
    ans = {}
    for matLikeImgInfo in matLikeImgInfos:
        if len(matLikeImgInfo) < 2:
            print("传入参数有误")
            continue
        imgName = matLikeImgInfo[0]
        matLikeImg = matLikeImgInfo[1]
        peopleName = config.kUnknownStr
        gray = cv2.cvtColor(matLikeImg, cv2.COLOR_BGR2GRAY)
        faceBox = kFaceCascade.detectMultiScale(gray)  # 获得脸部位置信息
        if not imgName in ans:
            ans[imgName] = []
        if len(faceBox) == 0:
            ans[imgName].append([peopleName, None])
            continue
        for index in range(len(faceBox)):
            (x, y, w, h) = faceBox[index]
            # goal越大越不相似
            id, goal = recognizer.predict(gray[y : y + h, x : x + w])
            if goal <= 50:
                peopleName = idToName.get(id)
            ans[imgName].append([peopleName, faceBox[index]])
    return ans
