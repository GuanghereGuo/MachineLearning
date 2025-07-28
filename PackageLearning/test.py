import os
from matplotlib.image import imread
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(script_dir, '..', 'DeepLearningFromScratchCode', 'dataset', 'lena_gray.png')
img = imread(image_path)
plt.imshow(img)

plt.show()