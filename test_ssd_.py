# -*- coding: utf-8 -*-
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
import cv2 as cv
import numpy as np
from  nets.ssd_net import SSD300
from utils.ssd_utils import BBoxUtility

class Video_tracker:
	def __init__(self,model,input_shape,name_classes):
		print ("カメラやビデオを処理する！")
		self.model = model
		self.num_classes = len(name_classes)
		self.name_classes = name_classes
		self.width,self.height=input_shape[0],input_shape[1]
		self.window_pos_x,self.window_pos_y=(60,40)
		self.bbox_util = BBoxUtility(num_classes=self.num_classes)
		
	def run(self,filepath,conf_thresh=0.4):
		"""
		"""
		frame = cv.imread(filepath)
		src = np.copy(frame)
		resized = cv.resize(frame,(self.width,self.height))
		rgb = cv.cvtColor(resized,cv.COLOR_BGR2RGB)
		src_shape = src.shape
		inputs = [img_to_array(rgb)]
		x= preprocess_input(np.array(inputs))
		y = self.model.predict(x)
		results = self.bbox_util.detection_out(y)
		to_draw= cv.resize(resized,(int(src_shape[1]),int(src_shape[0])))
			
		if len(results) > 0 and len(results[0]) > 0:
            # Interpret output, only one frame is used 
			det_label= results[0][:, 0]
			det_conf = results[0][:, 1]
			det_xmin = results[0][:, 2]
			det_ymin = results[0][:, 3]
			det_xmax = results[0][:, 4]
			det_ymax = results[0][:, 5]
			top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
			top_conf = det_conf[top_indices]
			top_label_indices = det_label[top_indices].tolist()
			top_xmin = det_xmin[top_indices]
			top_ymin = det_ymin[top_indices]
			top_xmax = det_xmax[top_indices]
			top_ymax = det_ymax[top_indices]

			for i in range(top_conf.shape[0]):
				xmin = int(round(top_xmin[i] * to_draw.shape[1]))
				ymin = int(round(top_ymin[i] * to_draw.shape[0]))
				xmax = int(round(top_xmax[i] * to_draw.shape[1]))
				ymax = int(round(top_ymax[i] * to_draw.shape[0]))
				class_num = int(top_label_indices[i])
				cv.rectangle(src, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
				text = self.name_classes[class_num] + " " + ('%.2f' % top_conf[i])
				text_top = (xmin, ymin-10)
				text_bot = (xmin + 80, ymin + 5)
				text_pos = (xmin + 5, ymin)
				cv.rectangle(src, text_top, text_bot, (0,255,0), -1)
				cv.putText(src, text, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

		cv.imshow("pic",src)
		cv.waitKey(0)
		cv.destroyAllWindows()


def main():
	print("Text".center(20,"_"))
	input_shape = (300,300,3)
	class_names = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
	NUM_CLASSES = len(class_names)
	model = SSD300(input_shape, num_classes=NUM_CLASSES)

	model.load_weights('ckpt/fine_tuning/weights.epoch_99--val_acc_0.64.hdf5') 
	obj = Video_tracker(model,input_shape,class_names )
	obj.run("image/2007_000876.jpg",0.4)

if __name__ == "__main__":
	main()
 
