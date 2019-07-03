import os
import xml.etree.ElementTree as ET


testsplit="./data/ImageSplits/test.txt"
trainsplit="./data/ImageSplits/train.txt"
annRoot="./data/XMLAnnotations/"

def update_text_util():
    new=open("new_train.txt","w")
    with open(trainsplit,"r") as f:
        for line in f:
            fname=line.strip()
            new.write(str(fname+extract_annotations(fname,annRoot)+"\n"))
    
        




def build_label_dict(label_src):
    label_dict={}
    c=0
    with open(str(label_src),"r") as f:
        for line in f:
            label_dict[line.rstrip()]=c
            c+=1
    return label_dict

def return_index_for_label(label):
    label_dict=build_label_dict("labels.txt")
    return int(label_dict[label])


def extract_annotations(image_id,annRoot):
    res=""
    width,height=0,0
    for file in os.listdir(annRoot):
        if file.startswith(image_id.split('.')[0]) and file.endswith('.xml'):
            extract=file
    tree=ET.parse(str(annRoot+extract))
    root=tree.getroot()
    for obj in root.iter('object'):
        label_index=return_index_for_label(obj.find('action').text.strip())
        bbox=obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)
            bndbox.append(cur_pt)
        bndbox.append(label_index)
        for i in bndbox:
            res+=" "+str(i)
    return res




if __name__ =="__main__":
    update_text_util()