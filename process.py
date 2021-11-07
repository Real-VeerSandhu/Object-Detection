import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('./test-images/image2.jpg')
imgplot = plt.imshow(img)
plt.show()