import numpy as np 
import torch , random
import matplotlib.pyplot as plt 
from memory_profiler import profile

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, utils

fullmodel = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")

fullmodel2 = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
import gc

fullmodel.to('cuda')
fullmodel2.to('cuda')

predictor = SamPredictor(fullmodel2)
maskgen = SamAutomaticMaskGenerator(fullmodel)


# def get_all_labels(imgarray,boxes):
# 	# Boxes is a list of [xyxy,] list of tuples
# 	predictor.set_image(imgarray)
# 	input_boxes = torch.tensor(boxes, device=predictor.device)
# 	transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, imgarray.shape[:2])
# 	masks, scores, logits = predictor.predict_torch(point_coords=None,point_labels=None,boxes=transformed_boxes,multimask_output=False,)
# 	return masks,scores, logits, None

def get_all_labels(imgarray, boxes):
	if imgarray is None or len(boxes) == 0:
		raise ValueError("Invalid image array or boxes.")

	predictor.set_image(imgarray)
	input_boxes = torch.tensor(boxes, device=predictor.device)

	# Transforming boxes
	transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, imgarray.shape[:2])
	masks, scores, logits = predictor.predict_torch(point_coords=None, point_labels=None,
													boxes=transformed_boxes, multimask_output=False)
	
	return masks, scores, logits, None


def sample_points_from_box(box,n=5,maxperbox=15): #5
	# sample randint
	# Will choose 10 random points in box if produces more
	pts = []
	lbls = []
	xrand = np.random.randint(box[0],box[2],size=n)
	yrand = np.random.randint(box[1],box[3],size=n)
	for x,y in zip(xrand,yrand):
		pts.append(np.array([x,y]))
		lbls.append(1)
	lenpoints = len(pts)
	if lenpoints>maxperbox:
		rs = random.sample(range(lenpoints),maxperbox)
		newpts = []
		newlbls = []
		for i in rs:
			newpts.append(pts[i])
			newlbls.append(lbls[i])
		pts = newpts
		lbls = newlbls

	return pts,lbls

def sample_negative_points(imgarray,boxes,maxpoints=500):
	x = np.ones((imgarray.shape[0],imgarray.shape[1]),dtype=np.bool_)
	for [xmin,ymin,xmax,ymax] in boxes:
		x[ymin:ymax,xmin:xmax] = False
	# # sample coordinates of maxpoints from x which are False	
	
	rs,cs = np.nonzero(x)
	pts = []
	lbls = []
	for x,y in zip(cs,rs):
		pts.append(np.array([x,y]))
		lbls.append(0)
	lenpoints = len(pts)
	if len(pts)>maxpoints:
		rts = random.sample(range(lenpoints),maxpoints)
		newpts = []
		newlbls = []
		for i in rts:
			newpts.append(pts[i])
			newlbls.append(lbls[i])
		pts = newpts
		lbls = newlbls
	return pts,lbls


def prompt_using_points(imgarray,boxes,maxboxes = 400):#400
	# Boxes is a list of [xyxy,] list of tuples
	# Will choose 20 random boxes if too many boxes
	sampledboxes = boxes
	if len(boxes)>maxboxes:
		sampledboxes = random.sample(boxes,maxboxes)

	allpoints = []
	point_labels = []

	for bnum,box in enumerate(sampledboxes):
		pts,lbls = sample_points_from_box(box)
		allpoints.extend(pts)
		finlabels = list(map(lambda x: x*bnum,lbls))
		point_labels.extend(lbls)
	
	### Negative points
	ptsneg, lblsneg = sample_negative_points(imgarray,boxes,maxpoints=int(len(allpoints)/1.5))
	allpoints.extend(ptsneg)
	finlabels.extend(lblsneg)
	point_labels.extend(lblsneg)
	###

	allpoints = np.array(allpoints)
	point_labels = np.array(point_labels)
	predictor.set_image(imgarray)
	print(allpoints.shape,point_labels.shape)
	masks, scores, logits = predictor.predict(point_coords=allpoints,point_labels=point_labels,multimask_output=False,)
	#import ipdb; ipdb.set_trace()
	torch.cuda.empty_cache()
	return masks,scores, logits,allpoints


def amg_using_points(imgarray,boxes):
	# Boxes is a list of [xyxy,] list of tuples
	H,W,_  = imgarray.shape
	allpoints = []
	point_labels = []
	for bnum,box in enumerate(boxes):
		pts,lbls = sample_points_from_box(box)
		allpoints.extend(pts)
		finlabels = map(lambda x: x*bnum,lbls)
		point_labels.extend(lbls)
	

	allpoints = np.array(allpoints)
	point_labels = np.array(point_labels)
	grid = []
	for pt in allpoints:
		x = pt[0]/W 
		y = pt[1]/H 
		grid.append(np.array([x,y]))
	grid = np.array(grid)
	#amg = SamAutomaticMaskGenerator(fullmodel)
	#amg = SamAutomaticMaskGenerator(fullmodel,points_per_side=None,point_grids=[grid])
	amg = SamPromptedAutomaticMaskGenerator(fullmodel,prompt_points=[grid])
	annots = amg.generate(imgarray)
	opboxes = []
	for annot in annots:
		obb = annot['bbox']
		opboxes.append([obb[0],obb[1],obb[0]+obb[2],obb[1]+obb[3]])
	return annots,opboxes

def amg_using_prompted_points(imgarray,rs,cs,H,W,sample_ratio=0.0005,maxpoints=5000):
	sm_size = len(rs)
	yn = np.random.choice(a=[False,True],size=sm_size,p=[1.0-sample_ratio,sample_ratio])
	allpoints = []
	point_labels = []
	for (y,x,sampleorno) in zip(rs,cs,yn):
			if sampleorno == False:
				continue
			allpoints.append(np.array([x/W,y/H]))
			point_labels.append(1)
	initpointlen = len(allpoints)
	if initpointlen > maxpoints:
		mxidcs = random.sample(range(initpointlen),maxpoints)
		newallpoints = []
		new_point_labels = []
		for idc in mxidcs:
			newallpoints.append(allpoints[idc])
			new_point_labels.append(point_labels[idc])
		allpoints = newallpoints
		point_labels = new_point_labels
	print(len(allpoints),len(point_labels),imgarray.shape)
	grid = np.array(allpoints)
	amg = SamAutomaticMaskGenerator(fullmodel,points_per_side=None,point_grids=[grid])
	annots = amg.generate(imgarray)
	opboxes = []
	for annot in annots:
		obb = annot['bbox']
		opboxes.append([obb[0],obb[1],obb[0]+obb[2],obb[1]+obb[3]])
	return annots,opboxes	
	
def amg_using_prompted_points_outside_aoi(imgarray,actualrs,actualcs,H,W,aoi,sample_ratio=0.5,maxpoints=5000):
	# aoi should be xmin,xmax,ymin,ymax format
	rs = []
	cs = []
	aoixmin = aoi[0]
	aoiymin = aoi[2]
	aoixmax = aoi[1]
	aoiymax  = aoi[3]
	for y,x in zip(actualrs,actualcs):
		if y < aoiymin or x > aoixmax or y >aoiymax or x<aoixmin:
			rs.append(y)
			cs.append(x)
	sm_size = len(rs)
	yn = np.random.choice(a=[False,True],size=sm_size,p=[1.0-sample_ratio,sample_ratio])
	allpoints = []
	point_labels = []
	for (y,x,sampleorno) in zip(rs,cs,yn):
			if sampleorno == False:
				continue
			allpoints.append(np.array([x/W,y/H]))
			point_labels.append(1)
	initpointlen = len(allpoints)
	if initpointlen > maxpoints:
		mxidcs = random.sample(range(initpointlen),maxpoints)
		newallpoints = []
		new_point_labels = []
		for idc in mxidcs:
			newallpoints.append(allpoints[idc])
			new_point_labels.append(point_labels[idc])
		allpoints = newallpoints
		point_labels = new_point_labels
	print(len(allpoints),len(point_labels),imgarray.shape)
	grid = np.array(allpoints)
	amg = SamAutomaticMaskGenerator(fullmodel,points_per_side=None,point_grids=[grid])
	annots = amg.generate(imgarray)
	opboxes = []
	for annot in annots:
		obb = annot['bbox']
		opboxes.append([obb[0],obb[1],obb[0]+obb[2],obb[1]+obb[3]])
	return annots,opboxes	

def show_boxes(boxes, ax):
	
	for box in boxes:
		x0, y0 = box[0], box[1]
		w, h = box[2] - box[0], box[3] - box[1]
		ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=20))


def show_mask(mask, ax, random_color=False):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
		color = np.array([30/255, 144/255, 255/255, 0.6])
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)

def view_image_transforms(imgarray,boxes,op,opfile):
	fig, axs = plt.subplots(nrows=1, ncols=1+len(boxes),figsize=(200,100))
	axs[0].axis('off')
	axs_img = axs[0]
	axs_img.imshow(imgarray)
	axs_img.set_title("prompt boxes")
	show_boxes(boxes,axs_img)
	for i in range(1,len(boxes)+1):
		mask = op[i-1,:,:,:]
		axs[i].axis('off')
		axs_img = axs[i]
		axs_img.set_title("generated mask")
		show_mask(mask,axs_img)
	plt.savefig(opfile)

def view_pt_transforms(imgarray,boxes,op,opfile):

	fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(200,100))
	axs[0].axis('off')
	axs_img = axs[0]
	axs_img.imshow(imgarray)
	axs_img.set_title("prompt boxes")
	show_boxes(boxes,axs_img)
	axs[1].axis('off')
	axs_img = axs[1]
	#axs_img.imshow(imgarray)
	axs_img.set_title("all objects mask")
	show_mask(op,axs_img)
	plt.savefig(opfile)
#@profile
def view_pt_amg_transforms(imgarray,ipboxes,opboxes,opfile):
	#import ipdb;ipdb.set_trace()
	fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(200,100))
	axs[0].axis('off')
	axs_img = axs[0]
	axs_img.imshow(imgarray)
	axs_img.set_title("prompt boxes")
	show_boxes(ipboxes,axs_img)
	axs[1].axis('off')
	axs_img = axs[1]
	axs_img.imshow(imgarray)
	axs_img.set_title("pseudolabels")
	show_boxes(opboxes,axs_img)
	plt.savefig(opfile)
	plt.close(fig)




def view_mask_amg_transforms(imgarray, step1op, step2boxes, opfile):
	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(200, 100))
	axs[0].axis('off')
	axs_img = axs[0]
	axs_img.imshow(imgarray)
	axs_img.set_title("step1 mask")
	show_mask(step1op, axs_img)
	axs[1].axis('off')
	axs_img = axs[1]
	axs_img.imshow(imgarray)
	axs_img.set_title("step2 boxes")
	show_boxes(step2boxes, axs_img)
	plt.savefig(opfile)

 


import skimage.io, skimage, skimage.transform 
from bs4 import BeautifulSoup

def get_imgarray_and_boxes(imgfile,imgannotationsfile):
	# opens an image and its annotation in VOC format
	imgarray = skimage.io.imread(imgfile)
	fp = open(imgannotationsfile)
	markup = fp.read()
	fp.close()
	bs = BeautifulSoup(markup, "lxml-xml")
	boxes = []
	objs = bs.find_all("object")
	for obj in objs:
		boxes.append([int(obj.xmin.text),int(obj.ymin.text),int(obj.xmax.text),int(obj.ymax.text)])
	return(imgarray,boxes)


#### Consolidating the below logic into one or two functions

#@profile
def create_pseudolabels_based_on_boxes(imgarray,boxes,imgfile,aoi,visualize=True):
	# First step is to get an all objects mask using the boxes
	op,scores, logits, points = prompt_using_points(imgarray,boxes)
	if visualize:
		view_pt_transforms(imgarray,boxes,op,"all_objects_mask_"+imgfile)
	# Second step is to get a pseudo label using the all object masks
	opp = op.transpose(1,2,0).squeeze()
	oppshape = opp.shape
	newarr = np.copy(imgarray)
	newarr[np.logical_not(opp)] = [0,0,0]
	p_r, p_c = opp.nonzero()
	annots,opboxes = amg_using_prompted_points_outside_aoi(imgarray,p_r,p_c,oppshape[0],oppshape[1],aoi)
	rectboxes = 0
	rect_bboxes = []
	rect_labels=[]
	rect_centers = []
	visboxes = []
	for annot in annots:
		rect_area = annot["bbox"][2] * annot["bbox"][3]
		act_area = annot["area"]
		width = annot["bbox"][2]
		height = annot["bbox"][3]
		# XYWHC1C2X2Y2
		rectboxes += 1
		rect_bboxes.append([annot["bbox"][0],annot["bbox"][1],annot["bbox"][2],annot["bbox"][3],annot["bbox"][0]+annot["bbox"][2]/2,annot["bbox"][1]+annot["bbox"][3]/2,annot["bbox"][0]+annot["bbox"][2],annot["bbox"][1]+annot["bbox"][3]])
		rect_labels.append("object")
		rect_centers.append([annot["bbox"][0]+annot["bbox"][2]/2,annot["bbox"][1]+annot["bbox"][3]/2])
		visboxes.append((annot["bbox"][0],annot["bbox"][1],annot["bbox"][0]+annot["bbox"][2],annot["bbox"][1]+annot["bbox"][3]))
	rect_centers_array = np.array(rect_centers)
	if visualize:
		view_mask_amg_transforms(imgarray,opp,visboxes,"all_objects_"+imgfile)
	return rectboxes, rect_bboxes, rect_labels, rect_centers_array, visboxes , rect_centers,annots


### Post processing SAM boxes after AMG
from sklearn.neighbors import NearestNeighbors
pixel_cutoff = 2
extra_dist_margin = 1.5

# def postprocess_sam(imgarray,rectboxes, rect_bboxes, rect_labels, rect_centers_array, visboxes, rect_centers):
# 	imgshape = imgarray.shape
# 	#import ipdb; ipdb.set_trace()
# 	try:
# 		nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(rect_centers_array)
# 	except ValueError :
# 		try:
# 			nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(rect_centers_array)
# 		except ValueError :
# 			return rect_bboxes, rect_labels, imgshape
from sklearn.neighbors import NearestNeighbors

def postprocess_sam(imgarray, rectboxes, rect_bboxes, rect_labels, rect_centers_array, visboxes, rect_centers):
	imgshape = imgarray.shape
	
	# Check the number of samples in rect_centers_array
	n_samples = len(rect_centers_array)
	if n_samples == 0:
		return rect_bboxes, rect_labels, imgshape  # No samples to process
	
	# Set n_neighbors based on the number of samples
	n_neighbors = min(5, n_samples)  # Limit n_neighbors to the number of samples available

	# Try fitting the NearestNeighbors model
	try:
		nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rect_centers_array)
	except ValueError as e:
		print(f"Error fitting NearestNeighbors: {e}")
		return rect_bboxes, rect_labels, imgshape  # Return early if fitting fails
	
	# Perform the kneighbors search
	rect_distances, rect_indices = nbrs.kneighbors(rect_centers_array)
	
	# Continue with the rest of your processing logic here...
	
	return rect_bboxes, rect_labels, imgshape  # Modify as needed to return results


	#############################################3
	rect_distances, rect_indices = nbrs.kneighbors(rect_centers_array)
	elimset = set()
	for (rectnum,distances,indices) in zip(range(rectboxes),rect_distances,rect_indices):
		rect_self = rect_bboxes[rectnum]
		rect_self_width = rect_self[2]
		rect_self_height = rect_self[3]
		for (distance,index) in zip(distances,indices):
			if index==rectnum:
				continue # Self is useless calc
			rect_other = rect_bboxes[index]
			rect_other_width = rect_other[2]
			rect_other_height = rect_other[3]
			maxdist = extra_dist_margin * ( np.sqrt(rect_other_width**2+rect_other_height**2)/2 - np.sqrt(rect_self_width**2+rect_self_height**2)/2 )
			if maxdist<=0:
				continue
			if distance<=maxdist:
				#print(maxdist,distance)
				elimset.add(rectnum)

	superbox_bboxes = []
	superbox_labels=[]
	superbox_centers = []
	bboxes = []

	for (rectnum,(bbox,label,center)) in enumerate( zip( rect_bboxes, rect_labels, rect_centers ) ):
		if rectnum not in elimset:

			# Applying cutoff to remove very small boxes, if in output
			this_width = bbox[2]
			this_height = bbox[3]
			if this_width <= pixel_cutoff or this_height <= pixel_cutoff:
				continue


			superbox_bboxes.append(bbox)
			superbox_labels.append(label)
			superbox_centers.append(center)
			bboxes.append([bbox[0],bbox[1],bbox[6],bbox[7]])


	labels = superbox_labels

	return bboxes, labels, imgshape



# maxx=1024
# maxy = 1024
# fullwindow1 = np.zeros((maxx,maxy),dtype=np.bool8)
# fullwindow2 = np.zeros((maxx,maxy),dtype=np.bool8)
# def get_iou(x1min,x1max,y1min,y1max,x2min,x2max,y2min,y2max):
# 	global fullwindow1, fullwindow2
# 	fullwindow1.fill(False)
# 	fullwindow2.fill(False)
# 	fullwindow1[x1min:x1max,y1min:y1max] = True
# 	fullwindow2[x2min:x2max,y2min:y2max] = True
# 	intersection = np.logical_and(fullwindow1, fullwindow2)
# 	union = np.logical_or(fullwindow1, fullwindow2)
# 	iou_score = np.sum(intersection) / np.sum(union)
# 	return iou_score

import numpy as np

maxx = 1024
maxy = 1024
fullwindow1 = np.zeros((maxx, maxy), dtype=np.bool8)
fullwindow2 = np.zeros((maxx, maxy), dtype=np.bool8)

def get_iou(x1min, x1max, y1min, y1max, x2min, x2max, y2min, y2max):
	global fullwindow1, fullwindow2
	
	# Ensure that coordinates are integers
	x1min, x1max, y1min, y1max = map(int, [x1min, x1max, y1min, y1max])
	x2min, x2max, y2min, y2max = map(int, [x2min, x2max, y2min, y2max])
	
	# Reset the boolean arrays
	fullwindow1.fill(False)
	fullwindow2.fill(False)
	
	# Ensure indices are within the valid range
	x1min = max(0, x1min)
	x1max = min(maxx, x1max)
	y1min = max(0, y1min)
	y1max = min(maxy, y1max)

	x2min = max(0, x2min)
	x2max = min(maxx, x2max)
	y2min = max(0, y2min)
	y2max = min(maxy, y2max)

	# Mark the boxes in the binary masks
	fullwindow1[x1min:x1max, y1min:y1max] = True
	fullwindow2[x2min:x2max, y2min:y2max] = True
	
	# Calculate the intersection and union
	intersection = np.logical_and(fullwindow1, fullwindow2)
	union = np.logical_or(fullwindow1, fullwindow2)

	# Calculate IoU
	union_area = np.sum(union)
	
	if union_area == 0:
		iou_score = 0  # Or use np.nan to indicate undefined IoU
	else:
		iou_score = np.sum(intersection) / union_area
		
	return iou_score



from torchvision.utils import draw_bounding_boxes
def see_postprocessed_boxes(imgarray,bboxes,labels):
	bboxes = torch.as_tensor(bboxes)
	iarr = torch.as_tensor(imgarray).permute([2,0,1])
	oarr = draw_bounding_boxes(iarr,boxes=bboxes,labels=labels)
	opimg = oarr.permute([1,2,0]).numpy()
	skimage.io.imsave("postprocessed_"+imgfile,opimg)
#@profile
def dont_take_boxes_already_at_start(inpboxes,sampostprocessedboxes):
	# inpboxes need to be xyxy
	# sampostprocessedboxes need to be xyxy as well
	preexistingboxes = []
	centers = []
	for box in inpboxes:
		xcenter = (box[0]+box[2])/2.0
		ycenter = (box[1] + box[3]) / 2.0
		xwidth = (box[2]-box[0])
		yheight = (box[3]-box[1])
		centers.append([xcenter,ycenter])
		preexistingboxes.append((box,[xwidth,yheight]))
	centers_array = np.asarray(centers)



	prefilterboxes = []
	samboxcenters = []
	for box in sampostprocessedboxes:
		xcenter = (box[0]+box[2])/2.0
		ycenter = (box[1] + box[3]) / 2.0
		xwidth = (box[2]-box[0])
		yheight = (box[3]-box[1])
		samboxcenters.append([xcenter,ycenter])
		prefilterboxes.append((box,[xwidth,yheight]))

	try:
		nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(centers_array)
		samboxcenters_array = np.asarray(samboxcenters)
		rect_distances, rect_indices = nbrs.kneighbors(samboxcenters_array)
	except ValueError:
		return [p[0] for p in prefilterboxes],["couldnt seprate boxes" for p in prefilterboxes]


	elimset = set()
	for i in range(len(prefilterboxes)):
		sambox = prefilterboxes[i][0] 
		nearindices = rect_indices[i]
		neardistances  = rect_distances[i]
		for (index,distance) in zip(nearindices,neardistances):
			actbox = preexistingboxes[index][0]
			iou = get_iou(sambox[0],sambox[2],sambox[1],sambox[3],actbox[0],actbox[2],actbox[1],actbox[3])
			if iou > 0.5:
				elimset.add(i)

	filteredsamboxes = []
	finlabels = []
	for i,(box,_) in enumerate(prefilterboxes):
		if not (i in elimset):
			filteredsamboxes.append(box)
			finlabels.append("object")
	
	return filteredsamboxes,finlabels



if __name__=="__main__":
	imgfile = "pseudo_shelf.jpg"
	imgannotations = "pseudo_shelf.xml" # Pascal VOC format
	imgarray,boxes = get_imgarray_and_boxes(imgfile,imgannotations)
	rectboxes, rect_bboxes, rect_labels, rect_centers_array, xyxyboxes, rect_centers = create_pseudolabels_based_on_boxes(imgarray,boxes,imgfile)
	bboxes, labels, imgshape = postprocess_sam(imgarray,rectboxes,rect_bboxes,rect_labels,rect_centers_array,xyxyboxes,rect_centers)
	#see_postprocessed_boxes(imgarray,bboxes,labels)
	finboxes,finlabels = dont_take_boxes_already_at_start(boxes,bboxes)
	see_postprocessed_boxes(imgarray,finboxes,finlabels)



