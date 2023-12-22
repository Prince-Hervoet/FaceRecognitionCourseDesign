import config
import os
import numpy
import cv2

# 图片路径
kPhotosPath = config.kPhotosPath
kSolvedPhotosPath = config.kSolvedPhotosPath
kFaceCascade = config.kFaceCascade
kTrainDataPath = config.kTrainDataPath


def getPhotoDirName():
    children = os.listdir(kPhotosPath)
    ans = []
    for file in children:
        if os.path.isfile(os.path.join(kPhotosPath, file)):
            continue
        ans.append(file)
    return ans


def trainData():
    count = 1
    dirNames = getPhotoDirName()
    faceDataList = []
    faceIdList = []
    nameToid = {}
    targetPath = os.path.join(kTrainDataPath, config.kTrainDataFileName)
    for dirName in dirNames:
        currentDirPath = os.path.join(kSolvedPhotosPath, dirName + os.sep)
        fileNames = os.listdir(currentDirPath)
        for fileName in fileNames:
            matLikeImg = cv2.imread(os.path.join(currentDirPath, fileName), 0)
            faceDataList.append(matLikeImg)
            id = nameToid.get(dirName)
            if id == None:
                nameToid[dirName] = count
                id = count
                count += 1
            faceIdList.append(id)
    idToName = dict([val, key] for key, val in nameToid.items())
    numpy.save(os.path.join(kTrainDataPath, config.kIdToNameFileName), idToName)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faceDataList, numpy.array(faceIdList))
    recognizer.write(targetPath)