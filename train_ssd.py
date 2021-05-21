import pickle
from utils.detection_generate import Generator
from utils.ssd_utils import BBoxUtility
from nets.ssd_net import SSD300
from utils.ssd_losses import MultiboxLoss
import keras
from keras.callbacks import ModelCheckpoint,TensorBoard
#import tensorflow as tf
#import warnings
#warning.filterwarnings("ignore")
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print ("--------------->",physical_devices)
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8 #half of the memory
set_session(tf.Session(config=config))


#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
#config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)
#summary_op = tf.summary.merge_all()
#init = tf.initialize_all_variables()
#sess = tf.Session(config=config)


start_data = "./datasets/prior_boxes_ssd300.pkl"
data_path = "./datasets/image_data.pkl"
image_path = "./datasets/commodity/JPEGImages"
weight_path="./ckpt/pre_trained/weights_SSD300.hdf5"

class SSDTrain:
	def __init__(self,num_classes = 21, input_shape=(300,300,3),epochs=12):
		self.num_classes = num_classes
		self.batch_size = 4
		self.input_shape = input_shape
		self.epochs = epochs

		self.gt_path = data_path
		self.image_path = image_path 

		prior = pickle.load(open(start_data,"rb"))

		self.bbox_util =BBoxUtility(self.num_classes,prior)

		self.pre_trained = weight_path
		self.model = SSD300(self.input_shape,num_classes=self.num_classes)

	def get_detection_data(self):
		gt = pickle.load(open(self.gt_path,"rb"))
		#print (gt)

		name_keys = sorted(gt.keys())
		number = int(round(0.8*len(name_keys)))
		train_keys = name_keys[:number]
		#print ("start->",train_keys)
		val_keys =  name_keys[number:]

		bbox_util_ =BBoxUtility(self.num_classes,gt)


		gen = Generator(gt,self.bbox_util,self.batch_size,self.image_path,train_keys,val_keys,(self.input_shape[0],self.input_shape[1]),do_crop = False)


		return gen

	def init_model_param(self):
		self.model.load_weights(self.pre_trained,by_name=True)
		freeze =[
		"input_1","conv1_1","conv1_2","pool1",
		"conv2_1","conv2_2","pool2",
		"conv3_1","conv3_2","conv3_3","pool3"]

		for L in self.model.layers:
			if L.name in freeze:
				L.trainable = False

		return None

	def compile(self):
		"""
		编译模型
		"""
		#distribution = tf.contrib.distribute.MirroredStrategy()
		#self.model.compile(optimizer=keras.optimizers.Adam(),loss=MultiboxLoss(self.num_classes,neg_pos_ratio=2).compute_loss,distribution=distribution)

		lr=0.0000000001
		self.model.compile(optimizer=keras.optimizers.Adam(lr),loss=MultiboxLoss(self.num_classes,neg_pos_ratio=2).compute_loss,metrics=['accuracy'])
	def fit_generator(self,gen):
		callback=[
			ModelCheckpoint("./ckpt/fine_tuning/weights.epoch_{epoch:02d}--val_acc_{val_acc:.2f}.hdf5",
			monitor="val_acc",
			save_best_only=True,
			save_weights_only=True,
			mode="auto",
			period=100
				),
			TensorBoard(log_dir = "./graph",histogram_freq=1,write_graph=True,write_images=True)
		]


		
		self.model.fit_generator(
				    gen.generate(train=True), 
                	gen.train_batches,
                	self.epochs, 
               		callbacks = callback,
               		validation_data = gen.generate(train=False),
               		nb_val_samples = gen.val_batches
	       
	              
	                )
		


if __name__ == "__main__":
	Train = SSDTrain(num_classes=21, input_shape=(300,300,3))
	gen = Train.get_detection_data()
	Train.init_model_param()
	Train.compile()
	Train.fit_generator(gen)