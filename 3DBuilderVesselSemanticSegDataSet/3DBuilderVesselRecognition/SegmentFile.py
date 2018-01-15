import random
fin = open("./ImageSets/Segmentation/trainval.txt", 'r' )
f80out = open("./ImageSets/Segmentation/train.txt", 'w')
f20out = open("./ImageSets/Segmentation/val.txt", 'w')
for line in fin:
    r = random.random()
    if r < 0.80:
        f80out.write(line)
    else:
        f20out.write(line)
fin.close()
f80out.close()
f20out.close()
