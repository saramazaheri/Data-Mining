import scipy
import scipy.ndimage
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import matplotlib.image as mpimg

def make_square(im, min_size=100, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im
x=input("Enter Your Picture Direct : ")
x="test/"+x
test_image = Image.open(x)
new_image = make_square(test_image)
new_image.save(x)
a = image_data = mpimg.imread(x) 
# Import Image
a_np = np.array(a)
a_r = a_np[:,:,0]
a_g = a_np[:,:,1]
a_b = a_np[:,:,2]

def comp_2d(image_2d):
	cov_mat = image_2d - np.mean(image_2d , axis = 1)
	eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat)) 
	p = np.size(eig_vec, axis =1)
	idx = np.argsort(eig_val)
	idx = idx[::-1]
	eig_vec = eig_vec[:,idx]
	eig_val = eig_val[idx]
	numpc = 30 
	# This is Number Of Pricipal Components (Change And See Results.)
	if numpc <p or numpc >0:
		eig_vec = eig_vec[:, range(numpc)]
	score = np.dot(eig_vec.T, cov_mat)
	recon = np.dot(eig_vec, score) + np.mean(image_2d, axis = 1).T # Normalization
	recon_img_mat = np.uint8(np.absolute(recon)) 
	# control complex eigenvalues
	return recon_img_mat
a_r_recon, a_g_recon, a_b_recon = comp_2d(a_r), comp_2d(a_g), comp_2d(a_b)
recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon))
recon_color_img = Image.fromarray(recon_color_img)
recon_color_img.save("Result.jpg")
recon_color_img.show()

#Sara Mazaheri
#Data Mining
#PCA Algorithm