from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np

model = YOLO('D:/YoloV8/Product_Segmentation_YoloV8/my_model_2.pt')

results = model('D:/YoloV8/Product_Segmentation_YoloV8/data', save=True, conf=0.75)

cropped_images_dir = 'D:/YoloV8/Product_Segmentation_YoloV8/cropped_segments'
os.makedirs(cropped_images_dir, exist_ok=True)


for i, result in enumerate(results):
    image = cv2.imread(result.path)
    masks = result.masks
    for j, mask in enumerate(masks):
        mask = mask.data[0].numpy()
        segmentation_mask = mask
        segmentation_mask_resized = cv2.resize(segmentation_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary_mask = (segmentation_mask_resized > 0.5).astype(np.uint8)
        segmented_region = cv2.bitwise_and(image, image, mask=binary_mask)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_segmented_region = segmented_region[y:y+h, x:x+w]
            alpha_channel = binary_mask[y:y+h, x:x+w] * 255
            cropped_segmented_region_with_alpha = np.dstack((cropped_segmented_region, alpha_channel))
            cropped_image_path = os.path.join(cropped_images_dir, f'segmented_region_only_{i}_{j}.jpg')
            cv2.imwrite(cropped_image_path,  cropped_segmented_region_with_alpha)
            print(f'Saved cropped image: {cropped_image_path}')