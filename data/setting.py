print("Python Package 설치 : pandas, numpy, matplotlib, scipy, sklearn, seaborn, statsmodels")
print("R Version 확인 및 설치 : R 4.1.2")
print("R Package 설치  : tidyverse, caret, reshape2")

try:
    import pandas, numpy, matplotlib, scipy, sklearn, seaborn, statsmodels
except Exception as e:
    print("패키지 설치 필요", e)
else:
    print("패키지 설치되어 있음")

import os
import shutil

#작업방, 다운로드 폴더 내 파일/폴더 싹 제거
target_dir = [
    {"target_dir" : r"D:\작업방", "except" : ["텍스트분석", "텍스트 분석"]},
    {"target_dir" : r"D:\Downloads", "except" : [] },
    {"target_dir" : r"C:\Users\poscouser\Downloads", "except" : [] },
    {"target_dir" : r"C:\Users\poscoedu_ph\Downloads", "except" : [] }
]
for target in target_dir:
    files = os.listdir(target["target_dir"])
    for file in files:
        except_check = False
        for except_file in target["except"]:
            if except_file in file:
                except_check = True
                break
        if except_check is False:
            try:
                os.remove(target["target_dir"] + "/" + file)
                print("파일 제거완료 : {}".format(target["target_dir"] + "\\" + file))
            except:
                try:
                    shutil.rmtree(target["target_dir"] + "/" + file)
                    print("폴더 제거완료 : {}".format(target["target_dir"] + "\\" + file))
                except:
                    print("!!!제거실패!!! : {}".format(target["target_dir"] + "\\" + file))

#사용자 폴더 내 파일 싹 제거
except_files = [
    ".idlerc",
    ".ipynb_checkpoints",
    ".ipython",
    ".jupyter",
    ".keras",
    ".matplotlib",
    ".vscode",
    "3D Objects",
    "ansel",
    "AppData",
    "Application Data",
    "Contacts",
    "Cookies",
    "Desktop",
    "Documents",
    "Downloads",
    "Favorites",
    "ief",
    "IntelGraphicsProfiles",
    "Links",
    "Local Settings",
    "Music",
    "My Documents",
    "NetHood",
    "NTUSER",
    "ntuser",
    "Pictures",
    "pip",
    "PrintHood",
    "Recent",
    "Saved Games",
    "Searches",
    "SendTo",
    "SPC files",
    "Videos",
    "시작 메뉴"
]
target_dir = [
    {"target_dir" : r"C:\Users\poscouser"},
    {"target_dir" : r"C:\Users\poscoedu_ph"}
]
for target in target_dir:
    files = os.listdir(target["target_dir"])
    for file in files:
        except_check = False
        for except_file in except_files:
            if except_file in file:
                except_check = True
                break
        if except_check is False:
            try:
                os.remove(target["target_dir"] + "/" + file)
                print("파일 제거완료 : {}".format(target["target_dir"] + "\\" + file))
            except:
                try:
                    shutil.rmtree(target["target_dir"] + "/" + file)
                    print("폴더 제거완료 : {}".format(target["target_dir"] + "\\" + file))
                except:
                    print("!!!제거실패!!! : {}".format(target["target_dir"] + "\\" + file))

#바탕화면 내 파일 싹 제거
except_files = [
    "desktop.ini",
    "iexplore.exe - 바로 가기.lnk",
    "로그아웃.lnk",
    "setting"
]
target_dir = [
    {"target_dir" : r"C:\Users\poscouser\Desktop"},
    {"target_dir" : r"C:\Users\poscoedu_ph\Desktop"}
]
for target in target_dir:
    files = os.listdir(target["target_dir"])
    for file in files:
        except_check = False
        for except_file in except_files:
            if except_file in file:
                except_check = True
                break
        if except_check is False:
            try:
                os.remove(target["target_dir"] + "/" + file)
                print("파일 제거완료 : {}".format(target["target_dir"] + "\\" + file))
            except:
                try:
                    shutil.rmtree(target["target_dir"] + "/" + file)
                    print("폴더 제거완료 : {}".format(target["target_dir"] + "\\" + file))
                except:
                    print("!!!제거실패!!! : {}".format(target["target_dir"] + "\\" + file))

#데이터 파일을 작업방으로 복사
for file in os.listdir():
    if file.split(".")[-1] == "csv":
        try:
            shutil.copy2(file, r"D:\작업방")
            print("데이터 파일 복사완료 : {}".format(file))
        except:
            print("!!!데이터 파일 복사실패!!! : {}".format(file))

#다른 파이썬 버전이 설치되어 있는지 체크
for file in os.listdir(r"c:\Users\poscouser\AppData\Local\Programs\Python"):
    if file != "Python38":
        print("!!!!! 파이썬 3.8 외 다른 버전 존재 : {} !!!!!".format(file))
    else:
        print("설치된 파이썬 버전 : {}".format(file))

os.system("pause")
