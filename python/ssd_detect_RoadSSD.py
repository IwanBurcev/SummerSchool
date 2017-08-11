#!/usr/bin/env python
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2
import argparse

# load label map
voc_labelmap_file = 'data/RoadSSD/labelmap_RoadSSD.prototxt'
# model
model_def = 'models/VGGNet/RoadSSD/deploy.prototxt'
model_weights = 'models/VGGNet/RoadSSD/VGG_RoadSSD_SSD_300x300_iter_12000.caffemodel'
# detection parameters
crop_size = 400
#smallcrop_size = 150
image_newsize = 800

file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def NMS(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
        c =  boxes[:,4]
        imgID = boxes[:,5]

        sames = np.zeros((len(imgID),len(imgID)), dtype=np.float)
        for i1 in xrange(len(imgID)):
          for i2 in xrange(len(imgID)):
            if imgID[i1]==imgID[i2]:
              sames[i1,i2]=0.0
            else:
              sames[i1,i2]=1.0
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(c)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
                same = sames[i,idxs[:last]]
		overlap = ((w * h) / area[idxs[:last]]) * same
                
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick]

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_list",
        help="Image list"
    )
    args = parser.parse_args()
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    oldchecksums=[]

    image_list = args.image_list
    image_list_file = open(image_list, 'r')

    for line in image_list_file:
#      filename = line.split("\n")[0]
      filename = line.split("\n")[0]
      image = caffe.io.load_image(filename)
  
      image_width = image.shape[1]
      image_height = image.shape[0]
  
      if image_width > image_height:
        scale = float(image_newsize)/float(image_height)
      else:
        scale = float(image_newsize)/float(image_width)
  
      new_image_height = int(image_height*scale)
      new_image_width = int(image_width*scale)

      image_resized = caffe.io.resize_image(image, (new_image_height, new_image_width))
  
      pyramid = []
      if new_image_width > new_image_height:
        pyramid.append([0,0,new_image_height,new_image_width]) # full image
        pyramid.append([0,0,image_newsize,image_newsize]) # left full square
        pyramid.append([0,new_image_width - image_newsize,image_newsize,new_image_width]) # right full square

        pyramid.append([0,0,crop_size,crop_size]) # top left crop
        pyramid.append([(new_image_height-crop_size)/2,0,(new_image_height+crop_size)/2,crop_size]) # central left crop
        pyramid.append([(new_image_height-crop_size),0,new_image_height,crop_size]) # bottom left crop

        p4 = 4*crop_size - new_image_width
        p3 = crop_size - p4

        pyramid.append([0,p4,crop_size,p4+crop_size]) # top left-center crop
        pyramid.append([(new_image_height-crop_size)/2,p4,(new_image_height+crop_size)/2,p4+crop_size]) # central left-center crop
        pyramid.append([(new_image_height-crop_size),p4,new_image_height,p4+crop_size]) # bottom left-center crop

        pyramid.append([0,2*p4+p3,crop_size,2*p4+p3+crop_size]) # top right-center crop
        pyramid.append([(new_image_height-crop_size)/2,2*p4+p3,(new_image_height+crop_size)/2,2*p4+p3+crop_size]) # central right-center crop
        pyramid.append([(new_image_height-crop_size),2*p4+p3,new_image_height,2*p4+p3+crop_size]) # bottom right-center crop

        pyramid.append([0,new_image_width-crop_size,crop_size,new_image_width]) # top right crop
        pyramid.append([(new_image_height-crop_size)/2,new_image_width-crop_size,(new_image_height+crop_size)/2,new_image_width]) # central right crop
        pyramid.append([(new_image_height-crop_size),new_image_width-crop_size,new_image_height,new_image_width]) # bottom right crop
        
        sc_w_iters = 0#new_image_width/smallcrop_size
        sc_h_iters = 0#new_image_height/smallcrop_size
        for sc_w in xrange(sc_w_iters):
          for sc_h in xrange(sc_h_iters):
            sc_x1 = sc_w*smallcrop_size
            sc_x2 = min(new_image_width,sc_w*smallcrop_size+smallcrop_size)
            sc_y1 = sc_h*smallcrop_size
            sc_y2 = min(new_image_height,sc_h*smallcrop_size+smallcrop_size)
            pyramid.append([sc_y1,sc_x1,sc_y2,sc_x2])
      else:
        pyramid.append([0,0,new_image_height,new_image_width])
        pyramid.append([0,0,image_newsize,image_newsize])
        pyramid.append([new_image_height - image_newsize,0,new_image_height,image_newsize])

        pyramid.append([0,0,crop_size,crop_size])
        pyramid.append([0,(new_image_width-crop_size)/2,crop_size,(new_image_width+crop_size)/2])
        pyramid.append([0,(new_image_width-crop_size),crop_size,new_image_width])

        p4 = 4*crop_size - new_image_height
        p3 = crop_size - p4

        pyramid.append([p4,0,p4+crop_size,crop_size])
        pyramid.append([p4,(new_image_width-crop_size)/2,p4+crop_size,(new_image_width+crop_size)/2])
        pyramid.append([p4,(new_image_width-crop_size),p4+crop_size,new_image_width])

        pyramid.append([2*p4+p3,0,2*p4+p3+crop_size,crop_size])
        pyramid.append([2*p4+p3,(new_image_width-crop_size)/2,2*p4+p3+crop_size,(new_image_width+crop_size)/2])
        pyramid.append([2*p4+p3,(new_image_width-crop_size),2*p4+p3+crop_size,new_image_width])

        pyramid.append([new_image_height-crop_size,0,new_image_height,crop_size])
        pyramid.append([new_image_height-crop_size,(new_image_width-crop_size)/2,new_image_height,(new_image_width+crop_size)/2])
        pyramid.append([new_image_height-crop_size,(new_image_width-crop_size),new_image_height,new_image_width])

        sc_w_iters = 0#new_image_width/smallcrop_size
        sc_h_iters = 0#new_image_height/smallcrop_size
        for sc_w in xrange(sc_w_iters):
          for sc_h in xrange(sc_h_iters):
            sc_x1 = sc_w*smallcrop_size
            sc_x2 = min(new_image_width,sc_w*smallcrop_size+smallcrop_size)
            sc_y1 = sc_h*smallcrop_size
            sc_y2 = min(new_image_height,sc_h*smallcrop_size+smallcrop_size)
            pyramid.append([sc_y1,sc_x1,sc_y2,sc_x2])
      print pyramid
      for i in xrange(len(pyramid)):
        if pyramid[i][0]<0:
          pyramid[i][0]=0
        if pyramid[i][1]<0:
          pyramid[i][1]=0
        if pyramid[i][2]>new_image_height:
          pyramid[i][2]=new_image_height
        if pyramid[i][3]>new_image_width:
          pyramid[i][3]=new_image_width
        if (pyramid[i][3]-pyramid[i][1])<150 or (pyramid[i][2]-pyramid[i][0])<150:
          pyramid[i][0]=0
          pyramid[i][1]=0
          pyramid[i][2]=new_image_height
          pyramid[i][3]=new_image_width
           
      boxes=[]
      print filename
      detfile=open(filename.split('.')[-2]+'_det.txt', 'w')
      for i in xrange(len(pyramid)):
        crop = pyramid[i]
        crop_scale_h=crop[2]-crop[0]
        crop_scale_w=crop[3]-crop[1]
        print str(new_image_height) + " "+str(new_image_width)+" " + str(crop[0])+" "+str(crop[1])+ " "+str(crop[2])+" "+str(crop[3])
        transformed_image = transformer.preprocess('data', image_resized[crop[0]:crop[2],crop[1]:crop[3],:])
        net.blobs['data'].data[...] = transformed_image
        # Forward pass.
        detections = net.forward()['detection_out']

        newchecksums=[]
        for z in xrange(detections.shape[2]):
          checksum=detections[0,0,z,2]+detections[0,0,z,3]+detections[0,0,z,4]+detections[0,0,z,5]+detections[0,0,z,6]
          newchecksums.append(checksum)

        for oldchecksum in oldchecksums:
          if oldchecksum not in newchecksums:
            del oldchecksum

        for z in xrange(detections.shape[2]):
          checksum=newchecksums[z]
          if checksum in oldchecksums:
            detections[0,0,z,2]=0
          else:
            oldchecksums.append(checksum)

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.1
        top_indices = [j for j, conf in enumerate(det_conf) if conf >= 0.2]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
    
        for j in xrange(len(top_conf)):
          if i==0 or (top_xmin[j]>0.01 and top_ymin[j]>0.01 and top_xmax[j]<0.99 and top_ymax[j]<0.99):
            boxes.append([top_xmin[j]*crop_scale_w+crop[1], top_ymin[j]*crop_scale_h+crop[0], top_xmax[j]*crop_scale_w+crop[1], top_ymax[j]*crop_scale_h+crop[0], top_conf[j], i,top_label_indices[j]])

      boxesArray = np.zeros((len(boxes), 7), dtype=np.float)
      for j in xrange(len(boxes)):
        boxesArray[j,:]=boxes[j][:]

      boxesNMSed = NMS(boxesArray, 0.3)

      img = Image.open(filename)
      dr = ImageDraw.Draw(img)

      COLOR_MAP = {
        "Low": (0, 200, 0),
        "Medium": (0, 225, 0),
        "High": (0, 255, 0)
      }
      line_width = 2

      for i in xrange(len(boxesNMSed)):
        xmin = int(boxesNMSed[i,0]/scale)
        ymin = int(boxesNMSed[i,1]/scale)
        xmax = int(boxesNMSed[i,2]/scale)
        ymax = int(boxesNMSed[i,3]/scale)
        score = boxesNMSed[i,4]
        label = get_labelname(voc_labelmap, int(boxesNMSed[i,6]))[0]
        if label=="tablet_case" and score<0.9:
          score = 0
        name = str(int(boxesNMSed[i,6]))+":"+label+":"+str(score) 
        print name+" "+str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)
        if score>0.9:
          color = COLOR_MAP["High"]
        if score>0.8 and score<=0.9:
          color = COLOR_MAP["Medium"]
        else:
          color = COLOR_MAP["Low"]
        try:
          detfile.write(str(xmin)+" "+str(ymin)+ " "+str(xmax)+" "+str(ymax) + " "+label + " "+str(score)+"\n")
          dr.line([xmin, ymin, xmax, ymin], fill=color, width=line_width)
          dr.line([xmax, ymin, xmax, ymax], fill=color, width=line_width)
          dr.line([xmin, ymax, xmax, ymax], fill=color, width=line_width)
          dr.line([xmin, ymin, xmin, ymax], fill=color, width=line_width)
          dr.text(((xmin + xmax)/2, ymin - 10),name, color)
        except:
          print filename, 'BAD'
      img.save('/home/eclipser/Data/RoadSSD/output/'+filename.split('/')[-1]+'_det.jpg')
      detfile.close()

if __name__ == "__main__":
    import sys
    main(sys.argv)
