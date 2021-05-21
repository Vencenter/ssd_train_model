import pickle
from utils.detection_generate import Generator
from utils.ssd_utils import BBoxUtility
from nets.ssd_net import SSD300
from utils.ssd_losses import MultiboxLoss
import keras
from keras.callbacks import ModelCheckpoint,TensorBoard
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


start_data = "./datasets/prior_boxes_ssd300.pkl"
data_path = "./datasets/image_data.pkl"
image_path = "./datasets/commodity/JPEGImages"


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8 #half of the memory
set_session(tf.Session(config=config))


class SSDTrain:
	def __init__(self,num_classes = 21, input_shape=(300,300,3),epochs=30):
		self.num_classes = num_classes
		self.batch_size = 1
		self.input_shape = input_shape
		self.epochs = epochs
		self.image_path = image_path

		self.gt_path = data_path
		self.model = SSD300(self.input_shape,num_classes=self.num_classes)
	



	def get_detection_data(self):
		gt = pickle.load(open(self.gt_path,"rb"))
		name_keys = sorted(gt.keys())
		number = int(round(0.8*len(name_keys)))
		train_keys = name_keys[:number]
		val_keys =  name_keys[number:]
		bbox_util_ =BBoxUtility(self.num_classes,gt)
		gen = Generator(bbox_util_,self.image_path,self.batch_size,train_keys,val_keys,(self.input_shape[0],self.input_shape[1]),num_classes=self.num_classes)
		return gen

	def compile(self):
		"""
		编译模型
		"""
		self.model.compile(optimizer=keras.optimizers.Adam(),loss=MultiboxLoss(self.num_classes,neg_pos_ratio=2).compute_loss,metrics=['accuracy'])

	def fit_generator(self,gen):
		callback=[
			ModelCheckpoint("./ckpt/fine_tuning/weights.{epoch:02d}--{val_acc:.2f}.hdf5",
			monitor="val_acc",
			save_best_only=True,
			save_weights_only=True,
			mode="auto",
			period=1
				),
			TensorBoard(log_dir = "./graph",histogram_freq=1,write_graph=True,write_images=True)
		]

		self.model.fit_generator(
				    gen.generate(train=True), 
                	gen.batch_size,
                	self.epochs, 
               		callbacks = callback,
               		validation_data = gen.generate(train=False),
               		nb_val_samples = gen.val_lines    )
		


if __name__ == "__main__":
	Train = SSDTrain(num_classes=21, input_shape=(300,300,3))
	gen = Train.get_detection_data()
	Train.compile()
	Train.fit_generator(gen)