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
            locateFaceAnalyse([fileName, matLikeImg], dirName)
    print("采样完成!")


# 从内存数据中截取人脸
# name 人名
def locateFaceFromData(matLikeImgInfo, name):
    util.clearOldPhotos(kSolvedPhotosPath)
    locateFaceAnalyse(matLikeImgInfo, name)


# matLikeImg MatLike类型的图片
# name 人名
def locateFaceAnalyse(matLikeImgInfo, name):
    if len(matLikeImgInfo) < 2:
        print("采样失败，传入参数有误")
        return
    imgName = matLikeImgInfo[0]
    matLikeImg = matLikeImgInfo[1]
    gray = cv2.cvtColor(matLikeImg, cv2.COLOR_BGR2GRAY)
    faceBox = kFaceCascade.detectMultiScale(gray)  # 获得脸部位置信息
    if len(faceBox) == 0:
        return
    dirName = name
    fileName = util.getUUIDStr() + "_" + imgName
    targetDirPath = os.path.join(kSolvedPhotosPath, dirName + os.sep)
    if not os.path.exists(targetDirPath):
        os.makedirs(targetDirPath)
    for x, y, w, h in faceBox:
        cv2.imwrite(
            os.path.join(targetDirPath, fileName),
            gray[y : y + h, x : x + w],
        )
    print("采样成功: " + dirName + " -- " + fileName)
