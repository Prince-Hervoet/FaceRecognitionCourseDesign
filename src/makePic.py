import config
import cv2
import os

# 图片路径
kPhotosPath = config.kPhotosPath
kSolvedPhotosPath = config.kSolvedPhotosPath
kFaceCascade = config.kFaceCascade


def getPhotoDirName():
    children = os.listdir(kPhotosPath)
    ans = []
    for file in children:
        if os.path.isfile(os.path.join(kPhotosPath, file)):
            continue
        ans.append(file)
    return ans


# 遍历数据集的图片，截取人脸
def locateFace():
    count = 1  # 简单计数
    dirNames = getPhotoDirName()
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
                    os.path.join(targetDirPath, str(count) + "_" + fileName),
                    gray[y : y + h, x : x + w],
                )
            count += 1
            print("采样成功: " + dirName + " -- " + fileName)