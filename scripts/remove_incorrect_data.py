import argparse
from PIL import Image
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

def png_only(file_list):
	return [f for f in file_list if f[-4:] == '.png']

root = args.data_path

kitchen_path = os.path.join(root, 'kitchen', 'images')
living_path = os.path.join(root, 'living', 'images')
bedroom_path = os.path.join(root, 'bedroom', 'images')
bathroom_path = os.path.join(root, 'bathroom', 'images')
#
kitchen_name_only = png_only(list(sorted(os.listdir(kitchen_path))))
living_name_only = png_only(list(sorted(os.listdir(living_path))))
bedroom_name_only = png_only(list(sorted(os.listdir(bedroom_path))))
bathroom_name_only = png_only(list(sorted(os.listdir(bathroom_path))))
#
kitchen = [os.path.join(kitchen_path, f) for f in kitchen_name_only]
living = [os.path.join(living_path, f) for f in living_name_only]
bedroom = [os.path.join(bedroom_path, f) for f in bedroom_name_only]
bathroom = [os.path.join(bedroom_path, f) for f in bathroom_name_only]
#
imgs = kitchen + living + bedroom + bathroom
masks = [f.replace("images/", "masks/") for f in imgs]
metas = [f.replace("images/", "meta/").replace(".png", ".json") for f in imgs]


identifiers_broken_img =[]
identifiers_broken_mask =[]
identifiers_broken_meta =[]

paths_broken_img =[]
paths_broken_mask =[]
paths_broken_meta =[]

counter = 0

for img_path, mask_path, meta_path in zip(imgs, masks, metas):
	counter +=1
	if counter %1000 == 0:
		print("counter is ", counter)
	identifier = img_path[0].split('/')[-1][:-4]
	#Try opening each of them 
	try:
		img = Image.open(img_path)
	except:
		paths_broken_img.append(img_path)

	mask = Image.open(mask_path)
	mask = np.array(mask)
	try:
		im_width, im_height = mask.shape[0], mask.shape[1]
	except:
		paths_broken_mask.append(mask_path)

	try:
		with open(meta_path, 'r') as f:
			color_to_object = json.load(f)
	except:
		paths_broken_meta.append(meta_path)

pickle.dump(paths_broken_img, open(os.path.join(args.data_path, "paths_broken_img.p") , "wb"))
pickle.dump(paths_broken_mask, open(os.path.join(args.data_path, "paths_broken_mask.p") , "wb"))
pickle.dump(paths_broken_meta, open(os.path.join(args.data_path, "paths_broken_meta.p") , "wb"))



