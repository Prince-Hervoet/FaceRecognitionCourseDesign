import shutil
import os
import uuid
import cv2
import config


# 清空文件夹
def clearOldPhotos(pathStr):
    shutil.rmtree(pathStr)
    os.mkdir(pathStr)


# 获取某个目录的子目录名称
def getPhotoDirName(pathStr):
    children = os.listdir(pathStr)
    ans = []
    for file in children:
        if os.path.isfile(os.path.join(pathStr, file)):
            continue
        ans.append(file)
    return ans


# 生成一个随机id
def getUUIDStr():
    return str(uuid.uuid1())


# 绘制人脸框
# name 人名
# faceBox 人脸位置信息
# sourceMatLikeImg 原始Matlike类型的图片
def drawFaceBox(name, oneFaceBox, sourceMatLikeImg):
    if oneFaceBox is None or len(oneFaceBox) == 0:
        return
    (x, y, w, h) = oneFaceBox
    cv2.rectangle(sourceMatLikeImg, (x, y), (x + w, y + h), color=(0, 0, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    textSize = cv2.getTextSize(name, font, 1, 2)
    cv2.rectangle(
        sourceMatLikeImg,
        (x, y - textSize[0][1] - textSize[1] - 1),
        (x + textSize[0][0], y),
        (0, 0, 255),
        -1,
    )
    cv2.putText(
        sourceMatLikeImg, name, (x, y - textSize[1]), font, 1, (255, 255, 255), 2
    )
    return sourceMatLikeImg


# 获取脸部信息
def getFace(matLikeImg):
    gray = cv2.cvtColor(matLikeImg, cv2.COLOR_BGR2GRAY)
    faceBox = config.kFaceCascade.detectMultiScale(gray)  # 获得脸部位置信息
    return faceBox


# 获取最大的人脸
def getMaxFaceBox(faceBox):
    if len(faceBox) < 2:
        return faceBox
    maxArea = 0
    ansIndex = 0
    for index, (x, y, w, h) in enumerate(faceBox):
        area = w * h
        if area > maxArea:
            maxArea = area
            ansIndex = index
    return [faceBox[ansIndex]]
