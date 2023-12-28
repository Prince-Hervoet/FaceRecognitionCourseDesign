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
            faceBox = locateFaceAnalyse(matLikeImg)
            if len(faceBox) == 0:
                continue
            fileName = util.getUUIDStr() + "_" + fileName
            targetDirPath = os.path.join(kSolvedPhotosPath, dirName + os.sep)
            if not os.path.exists(targetDirPath):
                os.makedirs(targetDirPath)
            for x, y, w, h in faceBox:
                cv2.imwrite(
                    os.path.join(targetDirPath, fileName),
                    matLikeImg[y : y + h, x : x + w],
                )
            print("采样成功: " + dirName + " -- " + fileName)
    print("采样完成!")


# 从内存数据中截取人脸
# name 人名
def locateFaceFromData(matLikeImg, name):
    util.clearOldPhotos(kSolvedPhotosPath)
    faceBox = locateFaceAnalyse(matLikeImg)
    if len(faceBox) == 0:
        return
    fileName = util.getUUIDStr() + "_" + fileName
    targetDirPath = os.path.join(kSolvedPhotosPath, name + os.sep)
    if not os.path.exists(targetDirPath):
        os.makedirs(targetDirPath)
    for x, y, w, h in faceBox:
        cv2.imwrite(
            os.path.join(targetDirPath, fileName),
            matLikeImg[y : y + h, x : x + w],
        )
    print("采样成功: " + name + " -- " + fileName)


# 定位人脸
# matLikeImg MatLike类型的图片
def locateFaceAnalyse(matLikeImg):
    gray = cv2.cvtColor(matLikeImg, cv2.COLOR_BGR2GRAY)
    faceBox = kFaceCascade.detectMultiScale(gray)  # 获得脸部位置信息
    if len(faceBox) == 0:
        return []
    # 获取最近的人脸
    faceBox = util.getMaxFaceBox(faceBox)
    return faceBox
