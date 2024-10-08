from sam_box_prompting4 import create_pseudolabels_based_on_boxes, postprocess_sam, dont_take_boxes_already_at_start, view_pt_amg_transforms
import numpy as np
import pandas as pd
import os
import skimage.io, skimage.transform
import shutil
import torch
import random
import gc
from sklearn.neighbors import NearestNeighbors
from skimage.io import imread,imsave
import psutil
import time
import gc
from memory_profiler import profile
import logging

batch_size = 10 

# Configure logging to file and console
logging.basicConfig(level=logging.INFO,
			   format='%(asctime)s - %(levelname)s - %(message)s',
			   handlers=[
				  logging.FileHandler("/datadisk/combined_dataset/pseudo_labeling.log"),  # Save log to file
				  logging.StreamHandler()  # logger.info log to console
			   ])

logger = logging.getLogger()

# Example usage
logger.info(f"Script started")



import pandas as pd

# Create an empty DataFrame to store image names and their bounding boxes
columns = ['imname', 'x_min', 'y_min', 'x_max', 'y_max']
pseudo_label_df = pd.DataFrame(columns=columns)

def save_pseudo_label(imname, boxes):
	"""
	Save image name and corresponding bounding boxes to the global DataFrame.
	
	:param imname: Name of the image.
	:param boxes: List of bounding boxes in the format [x_min, y_min, x_max, y_max].
	"""
	global pseudo_label_df
	#import ipdb; ipdb.set_trace()
	# Create a list to hold new rows
	new_rows = []

	# For each box, create a new row with the image name repeated
	for box in boxes:
		new_rows.append({
			'imname': imname,
			'x_min': box[0],
			'y_min': box[1],
			'x_max': box[2],
			'y_max': box[3]
		})

	# Create a DataFrame from the new rows and concatenate with the existing DataFrame
	new_df = pd.DataFrame(new_rows)
	pseudo_label_df = pd.concat([pseudo_label_df, new_df], ignore_index=True)

	# Log the operation
	logger.info(f"Saved pseudo-labels for {imname}")



# Set parameters
size_restriction = 1024#1800
groups_to_remove = ["Shelf", "Out Of Stock", "PriceTag", "Sticker", "OOS", "Barcode", "Sticker_Block"]
##@profile
def boxes_intersect(bbox1, bbox2, thresh=0.7):
	xmin = max(bbox1[0], bbox2[0])
	ymin = max(bbox1[1], bbox2[1])
	xmax = min(bbox1[2], bbox2[2])
	ymax = min(bbox1[3], bbox2[3])
	if xmax - xmin < 0 or ymax - ymin < 0:
		return False
	areas1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
	areas2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
	interA = (xmax - xmin) * (ymax - ymin)
	ov = float(interA) / min(areas1, areas2)
	return ov >= thresh



# def viz_pickle(product_boxes,imname):
# 	save_path = "/home/pdguest/shubham/viz/xyz.jpg"
	
# 	ori_im = imname
# 	if ori_im.shape[-1] == 4:
# 		ori_im = ori_im[:,:,0:3]
# 	ori_im = np.ascontiguousarray(ori_im)

# 	for gbox in product_boxes:
# 		ori_im = cv2.rectangle(ori_im, (int(gbox[0]),int(gbox[1])), (int(gbox[2]),int(gbox[3])), (255,0,0), 4)

# 	cv2.imwrite(save_path, ori_im)

##@profile
def filter_sam_only_boxes(annots,img_height,img_width):
	pixel_cutoff = 2#10
	rect_cutoff = 0.3
	shelf_avoid_width_height_ratio_cutoff = 8
	extra_dist_margin = 1.5
	img_width_height = 0.8
	rectboxes = 0
	rect_bboxes = []
	rect_scores=[]
	rect_centers = []
	
	for annot in annots:
		width = annot["bbox"][2]
		height = annot["bbox"][3]
		rect_area = width * height
		act_area = annot["area"]
		if width>=img_width*img_width_height or height>=img_height*img_width_height:
			continue
		if act_area<=rect_cutoff*rect_area:
			continue
		if width<pixel_cutoff or height<pixel_cutoff:
			continue # very small boxes escaped
		if width/height>shelf_avoid_width_height_ratio_cutoff:
			# Filtering wide stuff like shelves and price tags
			continue
		
		rectboxes += 1
		rect_bboxes.append([annot["bbox"][0],annot["bbox"][1],width,height,annot["bbox"][0]+width/2,annot["bbox"][1]+height/2,annot["bbox"][0]+width,annot["bbox"][1]+height])
		rect_scores.append(annot['stability_score'])
		rect_centers.append([annot["bbox"][0]+width/2,annot["bbox"][1]+height/2])
	
	if len(rect_centers)>10:
		rect_centers_array = np.array(rect_centers)
		nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(rect_centers_array)
		rect_distances, rect_indices = nbrs.kneighbors(rect_centers_array)
		elimset = set()
		for (rectnum,distances,indices) in zip(range(rectboxes),rect_distances,rect_indices):
			rect_self = rect_bboxes[rectnum]
			rect_self_width = rect_self[2]
			rect_self_height = rect_self[3]
			for (distance,index) in zip(distances,indices):
				if index==rectnum:
					continue 
				rect_other = rect_bboxes[index]
				rect_other_width = rect_other[2]
				rect_other_height = rect_other[3]
				intersection_flag = False
				if boxes_intersect(rect_self[:2]+rect_self[-2:], rect_other[:2]+rect_other[-2:]):
					intersection_flag = True
				maxdist = extra_dist_margin * ( np.sqrt(rect_other_width**2+rect_other_height**2)/2 - np.sqrt(rect_self_width**2+rect_self_height**2)/2 )
				if maxdist<=0:
					continue
				if distance<=maxdist and intersection_flag:
					
					elimset.add(rectnum)
		
		superbox_bboxes = []
		superbox_scores=[]
		superbox_centers = []
		bboxes = []
		for (rectnum,(bbox,score,center)) in enumerate( zip( rect_bboxes, rect_scores, rect_centers ) ):
			if rectnum not in elimset:
				superbox_bboxes.append(bbox)
				superbox_scores.append(score)
				superbox_centers.append(center)
				bboxes.append([bbox[0],bbox[1],bbox[6],bbox[7]])
		
		scores = superbox_scores
	else:
		
		bboxes = [xywh_to_xyxy(each['bbox']) for each in annots]
		scores = [each['stability_score'] for each in annots]
	return bboxes, scores

def xywh_to_xyxy(box):
	x, y, w, h = box
	return [x, y, x+w, y+h]
##@profile
def nms_inter_score(dets, scores, thresh=0.7):

	if len(dets) == 0:
		return []

	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		min_area = np.minimum(areas[i], areas[order[1:]])
		ovr = inter / (min_area)

		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]

	return keep



from PIL import Image
max_pixel_limit = 178956970 
Image.MAX_IMAGE_PIXELS = None 


logger.info(f"Processing started ")

# Set paths for CSV and image directory dynamically for each project

project_dir = "/datadisk/combined_dataset/final_project"
csv_fil = os.path.join(project_dir, "train_tag.csv")
image_directory = os.path.join(project_dir, "tagged_raw_images")

#viz_directory = os.path.join(project_dir, "viz")
viz_directory = "/datadisk/combined_dataset/final_project/viz"
# # Check if the viz folder exists and clear it
# if os.path.exists(viz_directory):
# 	shutil.rmtree(viz_directory)
# os.makedirs(viz_directory)


def is_valid_image_path(x):
	return isinstance(x, str) and not pd.isna(x)

chunk_size = 100000
chunk_number = 0
for chunk in pd.read_csv(csv_fil, chunksize=chunk_size):

	if chunk_number == 0:
		chunk_number += 1
		continue  # Skip the first chunk

	chunk = chunk[chunk["imname"].apply(is_valid_image_path)]

	chunk.loc[:, 'image_path'] = chunk["imname"].apply(lambda x: os.path.join(image_directory, os.path.basename(x)))


	grouped = chunk.groupby('image_path')
	images = list(grouped) 
	logger.info(f"currently running chunk number:{chunk_number}")
	# Increment the chunk counter
	chunk_number += 1

	

	for i in range(0, len(images), batch_size):
		batch_images = images[i:i + batch_size]
	# Process each grouped image
		total_images = len(batch_images) 
		processed_images = 0 
		skipped_images = []  
		for key, gdf in batch_images:
	
			try:
				logger.info(f"Working on filepath :: {key}")
				
				
				# Use PIL to get image dimensions without loading the entire image
				with Image.open(key) as img:
					H, W = img.size
				
				# Check if the image exceeds the pixel limit
				if H * W > max_pixel_limit:
					logger.info(f"Image {key} exceeds pixel limit. Skipping this image.")
					continue  
				else:
					# Load the image normally
					imgarray = skimage.io.imread(key)


				processed_images += 1

			except Exception as e:
				logger.info(f"Error processing {key}: {e}")
				skipped_images.append(key)  # Track images that caused errors
				continue  
			try :
			
				if not gdf.empty:  # Ensure gdf has data
					if gdf['a_ymin'].isnull().any() or gdf['a_ymax'].isnull().any() or gdf['a_xmin'].isnull().any() or gdf['a_xmax'].isnull().any():
						logger.info(f"Skipping image {key} due to NaN values in AOI.")
						continue

					minH = int(gdf['a_ymin'].min())
					maxH = int(gdf['a_ymax'].max())
					minW = int(gdf['a_xmin'].min())
					maxW = int(gdf['a_xmax'].max())

					image_area = H * W
					aoi_area = (maxH - minH) * (maxW - minW)

					if aoi_area >= image_area:
						logger.info(f"AOI covers the entire image {key}, skipping pseudo-labeling.")
						continue
				else:
					minH, maxH, minW, maxW = 0, H, 0, W

				newgdf = gdf[(gdf['x_min'] >= minW) & (gdf['x_max'] <= maxW) & (gdf['y_min'] >= minH) & (gdf['y_max'] <= maxH)]
				if len(newgdf) == 0:
					continue

				xyxy = [list(x) for x in zip(newgdf['x_min'], newgdf['y_min'], newgdf['x_max'], newgdf['y_max'])]
				imgname = newgdf["imname"].iloc[0].split("/")[-1]
				vizpath = os.path.join(viz_directory, imgname)

				newH, newW = None, None
				
				if H > size_restriction or W > size_restriction:
					if H > W:
						newH = size_restriction
						newW = int((size_restriction / H) * W)
						minW = int((size_restriction / H) * minW)
						maxW = int((size_restriction / H) * maxW)
						minH = int((size_restriction / H) * minH)
						maxH = int((size_restriction / H) * maxH)
						imgarray = skimage.transform.resize(imgarray, (newW, newH))
					else:
						newW = size_restriction
						newH = int((size_restriction / W) * H)
						minH = int((size_restriction / W) * minH)
						maxH = int((size_restriction / W) * maxH)
						minW = int((size_restriction / W) * minW)
						maxW = int((size_restriction / W) * maxW)
						imgarray = skimage.transform.resize(imgarray, (newW, newH))

					# Rescale bounding boxes accordingly
					
					xyxy = [[int(x[0] / W * newW), int(x[1] / H * newH), int(x[2] / W * newW), int(x[3] / H * newH)] for x in xyxy]

					#finboxes = [[int(x[0] * W / newW), int(x[1] * H / newH), int(x[2] * W / newW), int(x[3] * H / newH)] for x in finboxes]
	
				rectboxes, rect_bboxes, rect_labels, rect_centers_array, xyxyboxes, rect_centers, annots = create_pseudolabels_based_on_boxes(imgarray, xyxy, None, [minW, maxW, minH, maxH], False)
				img_height = imgarray.shape[0]
				img_width = imgarray.shape[1]
				
				filtered_bboxes, scores = filter_sam_only_boxes(annots, img_height, img_width)
		
				filtered_bboxes, scores = np.array(filtered_bboxes), np.array(scores)

				keep = nms_inter_score(filtered_bboxes, scores, thresh=0.7)

				filtered_bboxes= filtered_bboxes[keep]	
				
				finboxes, finlabels = dont_take_boxes_already_at_start(xyxy, filtered_bboxes)
				#import ipdb;ipdb.set_trace()
				finboxes1 = [[int(x[0] * W / newW), int(x[1] * H / newH), int(x[2] * W / newW), int(x[3] * H / newH)] for x in finboxes]
				save_pseudo_label(imgname, finboxes1)# need to resize again
				# #import ipdb;ipdb.set_trace()
				# viz_pickle(xyxy,imgarray)
				pseudo_label_df.to_csv('/datadisk/combined_dataset/pseudo_labels.csv', index=False)
		
				view_pt_amg_transforms(imgarray, xyxy, finboxes, vizpath)
			
				##import ipdb;ipdb.set_trace()
				
				logger.info(f'Length of rect_bboxess, filtered_bboxes, and finboxes respectively {len(rect_bboxes)}, {len(filtered_bboxes)}, {len(finboxes)}')


				# Cleanup
				del imgarray, rect_bboxes, filtered_bboxes, finboxes, gdf
				gc.collect()  # Call garbage collection

			except:
				continue

		torch.cuda.empty_cache()
		logger.info(f"Total images: {total_images}")
		logger.info(f"Successfully processed images: {processed_images}")
		logger.info(f"Skipped images (due to size limit or errors): {len(skipped_images)}")
		logger.info(f"Skipped image names: {skipped_images}")

	


logger.info(f"Processing completed.")





