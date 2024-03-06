import cv2
import os


PATH = "/home/speedy/Desktop/norm_RedditImageScraper-main/images"
img_list = os.listdir(PATH)

img_size = 128

port_count = 0
land_count = 0


print("Begin for loop!")
for img_f in img_list:
    if img_f.endswith(".png"):
        img = cv2.imread(os.path.join(PATH,img_f))
        h, w, c = img.shape
        if(h >= img_size and w >= img_size):
            h_ratio = img_size / h
            w_ratio = img_size / w  
            if(w_ratio >= h_ratio):
                port_count = port_count + 1
            else:
                land_count = land_count + 1

print("Finished for loop!")
cv2.destroyAllWindows()  

print("Portrait: ", port_count)
print("Landscape: ", land_count)
