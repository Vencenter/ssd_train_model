from xml.etree import ElementTree as ET 
from collections import OrderedDict as Dict
import numpy as np
import json
import pickle
import os



class xmlProcess(object):
	def __init__(self,file_path):
		self.xml_path = file_path
		self.num_class = 20
		self.data = Dict()

	def process_xml(self):
		for f in os.listdir(self.xml_path):
			#print (f)
			et = ET.parse(os.path.join(self.xml_path,f))
			root = et.getroot()
			size = root.find('size')

			width = eval(size.find("width").text)
			height =  eval(size.find("height").text)
			depth = eval(size.find("depth").text)

			bounding_boxes = []
			one_hots = []
			for obj in root.findall('object'):
				for res in obj.iter('bndbox'):
					xmin = float(res.find('xmin').text)/width
					ymin = float(res.find('ymin').text)/height
					xmax = float(res.find('xmax').text)/width
					ymax = float(res.find('ymax').text)/height

				bounding_boxes.append([xmin,ymin,xmax,ymax])
				obj_name = obj.find('name').text
				one_hot_name = self.one_hot(obj_name)
				one_hots.append(one_hot_name)
			image_name = root.find('filename').text
			bounding_boxes = np.asarray(bounding_boxes)
			one_hots  = np.asarray(one_hots)
			image_data = np.hstack((bounding_boxes,one_hots))
			self.data[image_name]=image_data 

		return None
	def one_hot(self,name):
                one_hot_vector = [0]*self.num_class

                if name == 'aeroplane':
                    one_hot_vector[0] = 1
                elif name == 'bicycle':
                    one_hot_vector[1] = 1
                elif name == 'bird':
                    one_hot_vector[2] = 1
                elif name == 'boat':
                    one_hot_vector[3] = 1
                elif name == 'bottle':
                    one_hot_vector[4] = 1
                elif name == 'bus':
                    one_hot_vector[5] = 1
                elif name == 'car':
                    one_hot_vector[6] = 1
                elif name == 'cat':
                    one_hot_vector[7] = 1
                elif name == 'chair':
                    one_hot_vector[8] = 1
                elif name == 'cow':
                    one_hot_vector[9] = 1
                elif name == 'diningtable':
                    one_hot_vector[10] = 1
                elif name == 'dog':
                    one_hot_vector[11] = 1
                elif name == 'horse':
                    one_hot_vector[12] = 1
                elif name == 'motorbike':
                    one_hot_vector[13] = 1
                elif name == 'person':
                    one_hot_vector[14] = 1
                elif name == 'pottedplant':
                    one_hot_vector[15] = 1
                elif name == 'sheep':
                    one_hot_vector[16] = 1
                elif name == 'sofa':
                    one_hot_vector[17] = 1
                elif name == 'train':
                    one_hot_vector[18] = 1
                elif name == 'tvmonitor':
                    one_hot_vector[19] = 1
                else:
                    print('unknown label: %s' %name)

                return one_hot_vector


if __name__ == "__main__":
	xp=xmlProcess("commodity/Annotations")
	xp.process_xml()
	#print (xp.data)
	pickle.dump(xp.data,open("./image_data.pkl","wb"))
	
	
