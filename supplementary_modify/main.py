# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:19:04 2020

@author: SC
"""

from trying import *
from test0 import *
import matplotlib as plt

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,'./dataset1','./dataset1/train_GT','train','SEG',data_gen_args,save_to_dir = None)
print(next(myGene)[0].shape)
print(next(myGene)[1].shape)
model_checkpoint = ModelCheckpoint('unet_dataset1.hdf5', monitor='loss',verbose=1, save_best_only=True)
model = unet()
#model = unet_test()
model.fit_generator(myGene,steps_per_epoch=500,epochs=10,callbacks=[model_checkpoint])

testGene = testGenerator("./dataset1/test")
results = model.predict_generator(testGene,30,verbose=1)

res1_path = './dataset1/res1'
if not osp.exists(res1_path):
    os.mkdir(res1_path)
    
for k in range(33):
    result = results[k]
    result = cv2.resize(result, (628,628))
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    result = result * 255
    result = unit16b2uint8(result)
    for i in range(628):
        for j in range(628):
            if result[i, j] == 255:
                color = np.random.randint(low=0, high=255, size=1)
                BFS(result, i, j, color)
    cv2.imwrite(osp.join(res1_path, 'mask{:0>3d}.tif'.format(k)), result.astype(np.uint16))

plt.imshow(results[0])
plt.show