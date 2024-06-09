import os
import xml.etree.ElementTree as ET
import shutil

txt_path=r'/home/zh/sxkai/new/iOD-main/data/NEU-DET/VOC2007/ImageSets/Main/source_train.txt'
xml_path=r'/home/zh/sxkai/new/iOD-main/data/NEU-DET/VOC2007/Annotations/'
a=[]
new_txt_path=r'/home/zh/sxkai/new/iOD-main/data/NEU-DET/VOC2007/ImageSets/Main/train.txt'
# class_name= [
# 'crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches'
# ]
# class_name= [
# 'defect', 'carriage', 'tee','branch','weld'
# ]
class_name= [
'crazing','inclusion','patches','pitted_surface'
]
# class_name= [
# 'weld', 'weld', 'weld','weld'
# ]
if os.path.exists(new_txt_path):
    with open(new_txt_path,'a+')as f:
        f.truncate(0)
with open(txt_path,'r')as f:
    lines=f.readlines()
    line=[line.rstrip().split(' ') for line in lines]
for item in line:
    name=item[0]+'.xml'
    tree=ET.parse(os.path.join(xml_path,name))
    root=tree.getroot()
    for name in root.iter('name'):
        a.append(name.text)
    if (class_name[0] not in a) and (class_name[1] not in a) and (class_name[2] not in a) and (class_name[3] not in a):
        with open(new_txt_path,'a+')as f:
            f.write(item[0]+'\n')
    a=[]
