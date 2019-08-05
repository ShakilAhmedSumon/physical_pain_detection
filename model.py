from keras import *
from keras.layers import *
from keras.regularizers import *
# from preprocess import Preprocess
import keras
import random
from keras import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os
import cv2
import numpy as np
from time import time
import time 


class ModelPainDetection():
	def __init__(self):
		
		self.train_set = np.load('train_image.npy')
		self.test_set = np.load('test_image.npy')
		self.y_train = np.load('y_train.npy')
		self.y_train = keras.utils.to_categorical(self.y_train)
		self.y_test = np.load('y_test.npy')
		self.y_test = keras.utils.to_categorical(self.y_test)
		


	def cnn_model(self):



		cnn_input = Input(shape = (160,160,3))

		conv = Conv2D(64, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(cnn_input)
		# conv = Dropout(.2)(conv)
		conv = Conv2D(32, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Conv2D(32, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dropout(.2)(conv)
		conv = Conv2D(16, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Conv2D(16, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Conv2D(16, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dropout(.2)(conv)
		conv = Flatten()(conv)
		conv = Dense(50, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dense(100, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dense(6, activation = 'softmax')(conv)

		model = Model(cnn_input, conv)

		return model


	def reshape_data(self):
		self.train_set = self.train_set.reshape((self.train_set.shape[0],40,32,1))
		self.test_set = self.test_set.reshape((self.test_set.shape[0],40,32,1))


	def data_generator(self):
		while True:
			random_number = random.randint(1, (43477-32))
			train_data = self.train_set[random_number:random_number+10]/255.0
			train_label = self.y_train[random_number:random_number+10]

			yield ({'input_1': train_data}, {'dense_3': train_label})



	def scheduler(self,epoch):
		if epoch < 10:
			return 0.001
		else:
			return 0.001 * tf.math.exp(0.1 * (10 - epoch))


	def run_model(self):

		# self.train_set = self.train_set/255.0
		self.test_set = self.test_set/255.0
		print(self.y_test[:100])
		

		model = self.cnn_model()
		# self.reshape_data()
		model.summary()

		filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period = 100)
		callbacks_list = [checkpoint]


		adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)

		model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
		# callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
		model.fit_generator(self.data_generator(), steps_per_epoch=500, epochs=500, validation_data = (self.test_set, self.y_test), verbose = 1,callbacks=callbacks_list)
		# model.save_weights('pain.h5')



	def inference(self):

		model = self.cnn_model()
		model.load_weights('./weigths/best_model.hdf5')
		model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

		for images in os.listdir('./images/'):
			image = cv2.imread('./images/' + images)
			image = cv2.resize(image, (160,160), interpolation = cv2.INTER_AREA)
			image = image.reshape(1,160,160,3)
			pred = model.predict(image)
			pred = np.asarray(pred)
			pain_scale = np.argmax(pred[0])
			if pain_scale < 3:
				print('Low pain')
			elif pain_scale == 3:
				print('mild pain')
			else:
				print('intense pain')
	def get_inference_from_videos(self):

		model = self.cnn_model()
		model.load_weights('./weigths/best_model.hdf5')
		model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

		cap = cv2.VideoCapture("./video_4.avi")
		while not cap.isOpened():
			cap = cv2.VideoCapture("./video_4.avi")
			cv2.waitKey(1000)
			print ("Wait for the header")

		pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
		while True:
			time.sleep(.01)
			flag, frame = cap.read()
			if flag:
				image = cv2.resize(frame, (160,160), interpolation = cv2.INTER_AREA)
				image = image.reshape(1,160,160,3)
				pred = model.predict(image)
				pred = np.asarray(pred)
				pain_scale = np.argmax(pred[0])
				if pain_scale < 3:
					# print('Low pain')
					cv2.putText(frame, str('LOW PAIN'),(5,50),0, 5e-3 * 200, (50,205,50),2)
				elif pain_scale == 3:
					cv2.putText(frame, str('MILD PAIN'),(5,50),0, 5e-3 * 200, (0,255,255),2)
				else:
					cv2.putText(frame, str('INTENSE PAIN'),(5,50),0, 5e-3 * 200, (0,0,255),2)

				frame = cv2.resize(frame, (640,480))
				cv2.imshow('video', frame)
				pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
				print (str(pos_frame)+" frames")
			else:
				cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
				cv2.waitKey(1000)

			if cv2.waitKey(10) == 27:
				break
			if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
				break


if __name__ == '__main__':
	pain = ModelPainDetection()
	pain.get_inference_from_videos()




