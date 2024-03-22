import cv2
import albumentations as A

PATH_IMAGE = "/home/enrico/Projects/Image_Anomaly_Detection/test_data_augmentation/002.png"


# used for v4 with 400 copies
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.3, p=0.6),
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=2, p=0.6),
    A.RandomGamma(p=0.6),
    A.Perspective(scale=(0.012, 0.012), p=0.6)
])

image = cv2.imread(PATH_IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
name_file_no_ext = PATH_IMAGE.split("/")[-1].split(".")[0]
file_ext = PATH_IMAGE.split("/")[-1].split(".")[-1]

for i in range(20):
    augmented_image = transform(image=image)['image']
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/home/enrico/Projects/Image_Anomaly_Detection/test_data_augmentation/" + name_file_no_ext + "_" + str(i+1) + "." + file_ext, augmented_image)