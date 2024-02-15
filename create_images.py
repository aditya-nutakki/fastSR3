import os, cv2
import numpy as np


lr_sz = 32

base_path = f"/mnt/d/work/datasets/celebA_gen/{lr_sz}_128"
reference_path = "/mnt/d/work/datasets/celebA_test/"

subdirs = ["10", "50", "100", "200", "500", "1000", "2000"]

# subdirs = [os.path.join(base_path, _subdir) for _subdir in os.listdir(base_path)]
subdirs = [os.path.join(base_path, _subdir) for _subdir in subdirs]

all_images = [os.listdir(subdir) for subdir in subdirs]
min_indices = [len(os.listdir(images)) for images in subdirs]
min_index = min_indices.index(min(min_indices))
print(min_index)
print(min_index, len(all_images[min_index]), subdirs[min_index])

minimum_images = all_images[min_index]
all_images.pop(min_index)

common_images =  []

def is_present(image, subdir_images):
    return True if image in subdir_images else False


for image in minimum_images:
    for other_images in all_images:
        is_common = is_present(image, other_images)
        if is_common == False:
            break
        
    if is_common:
        common_images.append(image.split("/")[-1])

# print(common_images, len(common_images))
        

def concat_images(stitched_image, axis = 1):
    return np.concatenate(stitched_image, axis = axis)


def resize(image, tgt_size = lr_sz):
    return cv2.resize(image, (tgt_size, tgt_size), interpolation = cv2.INTER_CUBIC)


def read_images(image_name):
    
    image_paths = [os.path.join(_subdir, image_name) for _subdir in subdirs]
    images = [cv2.imread(image) for image in image_paths]
    reference_image = os.path.join(reference_path, image_name)
    images.insert(0, resize(resize(cv2.imread(reference_image), lr_sz), 128))
    images.insert(len(images), resize(cv2.imread(reference_image), tgt_size=128))
    stitched_image = concat_images(images)
    display_images(image_name, stitched_image)


def save_image(image, image_name = None, mode = "good", save_path = None):
    print("saving")
    # cv2.imwrite(f"./32_128/{mode}/{image_name}", image)
    if save_path:
        print("jhere")
        cv2.imwrite(save_path, image)
    else:
        cv2.imwrite(f"./32_128/{mode}/{image_name}", image)


def display_images(image_name, stitched_image):
    cv2.imshow(image_name, stitched_image)


    key = cv2.waitKey(0)
    if key == ord('s'): 
        save_image(image_name, stitched_image, mode = "good")

    elif key == ord('b'): 
        save_image(image_name, stitched_image, mode = "bad")

    cv2.destroyAllWindows()


# for image in common_images:
#     read_images(image)
    

def stitch_images(path):
    image_paths = [os.path.join(path, image) for image in os.listdir(path)]
    images = [cv2.imread(image) for image in image_paths]
    stitched_image = concat_images(images, axis = 0)
    print(stitched_image.shape)
    save_image(stitched_image, save_path = os.path.join(path, "stitched_image.jpeg"))

stitch_images("/mnt/d/work/projects/pfs/sr3/images/32_128/bad")
