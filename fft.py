import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load gambar
img = Image.open('love.jpg').convert('L')
img_arr = np.asarray(img)

# Menghitung FFT dari gambar
fft_output = np.fft.fft2(img_arr)
fft_output_shifted = np.fft.fftshift(fft_output)
mag_spectrum = 20*np.log(np.abs(fft_output_shifted))

# Plot gambar asli dan spektrum frekuensi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(img, cmap='gray')
ax1.set_title('Gambar Asli')
ax1.axis('off')
ax2.imshow(mag_spectrum, cmap='gray')
ax2.set_title('Spektrum Frekuensi')
ax2.axis('off')
plt.show()
