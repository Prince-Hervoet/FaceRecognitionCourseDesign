import config
import cv2
import os
import util

# 图片路径
kPhotosPath = config.kPhotosPath
kSolvedPhotosPath = config.kSolvedPhotosPath
kFaceCascade = config.kFaceCascade

"""
matLikeImgInfo 是 [图片名称, MatLike类型的图片]
"""


# 遍历数据集的图片，截取人脸
def locateFaceFromFiles():
    print("采样中......")
    util.clearOldPhotos(kSolvedPhotosPath)
    dirNames = util.getPhotoDirName(kPhotosPath)
    for dirName in dirNames:
        currentDirPath = os.path.join(kPhotosPath, dirName + os.sep)
        fileNames = os.listdir(currentDirPath)
        for fileName in fileNames:
            matLikeImg = cv2.imread(os.path.join(currentDirPath, fileName))
            oneFaceBox = locateOneFace(matLikeImg)
            if oneFaceBox is None:
                continue
            fileName = util.getUUIDStr() + "_" + fileName
            targetDirPath = os.path.join(kSolvedPhotosPath, dirName + os.sep)
            if not os.path.exists(targetDirPath):
                os.makedirs(targetDirPath)
            (x, y, w, h) = oneFaceBox
            gray = cv2.cvtColor(matLikeImg, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(
                os.path.join(targetDirPath, fileName),
                gray[y : y + h, x : x + w],
            )
            print("采样成功: " + dirName + " -- " + fileName)
    print("采样完成!")


# 从内存数据中截取人脸
# name 人名
def locateFaceFromData(matLikeImg, name):
    oneFaceBox = locateOneFace(matLikeImg)
    if oneFaceBox is None:
        return
    fileName = util.getUUIDStr() + "_" + name + ".jpg"
    targetDirPath = os.path.join(kSolvedPhotosPath, name + os.sep)
    if not os.path.exists(targetDirPath):
        os.makedirs(targetDirPath)
    (x, y, w, h) = oneFaceBox
    gray = cv2.cvtColor(matLikeImg, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(
        os.path.join(targetDirPath, fileName),
        gray[y : y + h, x : x + w],
    )
    print("采样成功: " + name + " -- " + fileName)


# 定位一张脸
def locateOneFace(matLikeImg):
    faceBox = util.getFace(matLikeImg)
    faceBox = util.getMaxFaceBox(faceBox)
    if len(faceBox) == 0:
        return None
    return faceBox[0]
