from nets.resnet import get_resnet
from PIL import Image
import numpy as np
import os
import sys
apath = os.path.abspath(os.path.dirname(sys.argv[0]))


model = get_resnet(None,None,3)
model.load_weights(apath+"\\weights\\monet2photo\\g_BA_epoch0.h5")

img = np.array(Image.open(apath+"\\datasets\\monet2photo\\trainB\\2013-11-27 08_23_33.jpg").resize([256,256]))/127.5 - 1
img = np.expand_dims(img,axis=0)
fake = (model.predict(img)*0.5 + 0.5)*255

face = Image.fromarray(np.uint8(fake[0]))
face.show()