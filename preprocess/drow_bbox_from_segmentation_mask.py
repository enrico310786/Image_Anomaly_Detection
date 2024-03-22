import cv2


path_segmentation_mask = "/home/enrico/Projects/Image_Anomaly_Detection/preprocess/001_mask.png"
path_image = "/home/enrico/Projects/Image_Anomaly_Detection/preprocess/001.png"
path_image_bbox = "/home/enrico/Projects/Image_Anomaly_Detection/preprocess/001_bbox.png"


# Carica l'immagine e la segmentation mask
image = cv2.imread(path_image)
mask = cv2.imread(path_segmentation_mask, cv2.IMREAD_GRAYSCALE)

# Trova i contorni nella segmentation mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Determina il bounding box della segmentation mask
x, y, w, h = cv2.boundingRect(contours[0])

# Disegna il bounding box sull'immagine
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Mostra l'immagine con il bounding box
#cv2.imshow('Bounding Box', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#save the image
cv2.imwrite(path_image_bbox, image)