import config
import os
import numpy
import cv2
import util

# 图片路径
kPhotosPath = config.kPhotosPath
kSolvedPhotosPath = config.kSolvedPhotosPath
kFaceCascade = config.kFaceCascade
kTrainDataPath = config.kTrainDataPath


# 使用solvedPhotos中的图片进行训练，结果保存为xml和npy
def trainData():
    print("训练中......")
    util.clearOldPhotos(kTrainDataPath)
    dirNames = util.getPhotoDirName(kSolvedPhotosPath)
    count = 1
    faceDataList = []
    faceIdList = []
    nameToId = {}
    targetPath = os.path.join(kTrainDataPath, config.kTrainDataFileName)
    for dirName in dirNames:
        currentDirPath = os.path.join(kSolvedPhotosPath, dirName + os.sep)
        fileNames = os.listdir(currentDirPath)
        for fileName in fileNames:
            matLikeImg = cv2.imread(os.path.join(currentDirPath, fileName), 0)
            faceDataList.append(matLikeImg)
            id = nameToId.get(dirName)
            if id == None:
                nameToId[dirName] = count
                id = count
                count += 1
            faceIdList.append(id)
    idToName = dict([val, key] for key, val in nameToId.items())
    numpy.save(os.path.join(kTrainDataPath, config.kIdToNameFileName), idToName)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faceDataList, numpy.array(faceIdList))
    recognizer.write(targetPath)
    print("训练完成!")
