import numpy as np
import matplotlib.pyplot as plt

# Generate random numpy array with values from 0 to 255 and a size of 256 * 256
random_image = np.random.randint(0, 256, (256, 256))

plt.figure(figsize=(7, 7))
plt.imshow(random_image, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()