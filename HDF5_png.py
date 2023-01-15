import h5py
import os
from PIL import Image
import numpy as np
hdf5_path = r'./testSet_c7_ver_2.hdf5'  # 读取路径
store_path = r'./data'  # 存储路径
HDF5File = h5py.File(hdf5_path)
if not os.path.exists(store_path):
    os.mkdir(store_path)
images = HDF5File['images']['images'][:]
images = images.reshape(images.shape[0], 224, 224, 3)
labels=HDF5File['labels']['labels'][:]
label_names=['asphalt','grass','gravel','pavement','sand','brick','coated floor']
for i in range(images.shape[0]):
    img=images[i] # 表示第i张图片
    label=labels[i] # 表示第i张图片的标签
    label_name=label_names[label]
    path=store_path+'/'+label_name
    if not os.path.exists(path):
        os.mkdir(path)
    name=os.path.join(path,str(i)+'.png')
    img1=Image.fromarray(np.uint8(img))
    img1.convert('RGB').save(name)
    print('img'+str(i)+'is done!')
