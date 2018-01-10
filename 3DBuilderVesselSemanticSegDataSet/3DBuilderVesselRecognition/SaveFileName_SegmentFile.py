import os
thefile = open('./ImageSets/Segmentation/trainval.txt', 'w')
dir_path = "./JPEGImages"
files = [f for f in os.listdir('./JPEGImages') if os.path.isdir(dir_path)]
for f in files:
    temp = f.split('.')
    thefile.write("%s\n" % temp[0])
thefile.close()

cmd = 'python ./SegmentFile.py'
os.system(cmd)

