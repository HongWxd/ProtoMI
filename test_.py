import os 

path = '/data/hwx/boron/patents/'
files_path = os.listdir(path)

files = [i for i in files_path if i.endswith('.pdf')]
print(len(files))