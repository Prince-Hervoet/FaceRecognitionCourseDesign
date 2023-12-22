import config
import cv2
import os
import util

# 图片路径
kPhotosPath = config.kPhotosPath
kSolvedPhotosPath = config.kSolvedPhotosPath
kFaceCascade = config.kFaceCascade


# 遍历数据集的图片，截取人脸
def locateFace():
    util.clearOldPhotos(kSolvedPhotosPath)
    dirNames = util.getPhotoDirName(kPhotosPath)
    for dirName in dirNames:
        currentDirPath = os.path.join(kPhotosPath, dirName + os.sep)
        targetDirPath = os.path.join(kSolvedPhotosPath, dirName + os.sep)
        fileNames = os.listdir(currentDirPath)
        for fileName in fileNames:
            matLikeImg = cv2.imread(os.path.join(currentDirPath, fileName))
            gray = cv2.cvtColor(matLikeImg, cv2.COLOR_BGR2GRAY)
            faceBox = kFaceCascade.detectMultiScale(gray)  # 获得脸部位置信息
            if len(faceBox) == 0:
                continue
            if not os.path.exists(targetDirPath):
                os.makedirs(targetDirPath)
            for x, y, w, h in faceBox:
                cv2.imwrite(
                    os.path.join(targetDirPath, fileName),
                    gray[y : y + h, x : x + w],
                )
            print("采样成功: " + dirName + " -- " + fileName)
