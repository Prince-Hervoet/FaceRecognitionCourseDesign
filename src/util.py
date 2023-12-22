import shutil
import os


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
