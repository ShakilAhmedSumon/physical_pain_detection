import cv2
import os
import numpy as np 
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import tqdm

class Preprocess():
	def __init__(self,image_dir,label_dir):
		self.image_dir = image_dir
		self.label_dir = label_dir

	def extract_face(self,filename,required_size=(160, 160)):
		image = Image.open(filename)
		image = image.convert('RGB')
		pixels = asarray(image)
		detector = MTCNN()
		results = detector.detect_faces(pixels)
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		face = pixels[y1:y2, x1:x2]
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)
		return face_array

	def read_images_and_labels(self):
		train_image = []
		test_image = []
		train_labels = []
		test_labels = []

		for image_folder in os.listdir(self.image_dir):
			for images in os.listdir(self.image_dir + image_folder + '/'):
				f = open(self.label_dir + image_folder + '/' + images + '.txt', 'r')
				labels = f.read()
				i = 0
				for image in os.listdir(self.image_dir + image_folder + '/' + images + '/'):


					image_data = cv2.imread(self.image_dir + image_folder + '/' + images + '/' + image, cv2.IMREAD_UNCHANGED)
					print(self.image_dir + image_folder + '/' + images + '/' + image)
					if image[-4:] == '.png':
						image_data = cv2.resize(image_data, (160,160), interpolation = cv2.INTER_AREA)
						print(image_data.shape)
						
						if i%10 == 0:
							test_image.append(image_data)
							test_labels.append(labels)
						else:
							train_image.append(image_data)
							train_labels.append(labels)

						i = i + 1
		train_image = np.asarray(train_image)
		test_image = np.asarray(test_image)
		train_labels = np.asarray(train_labels)
		test_labels = np.asarray(test_labels)

		np.save('train_image.npy', train_image)
		np.save('test_image.npy', test_image)
		np.save('train_labels.npy', train_labels)
		np.save('test_labels.npy', test_labels)


	def get_labels_from_files(self):
		train_labels_npy = np.load('train_labels.npy')
		test_labels_npy = np.load('test_labels.npy')
		train_labels = []
		test_labels = []

		for label in train_labels_npy:
			train_labels.append(int(label[3]))

		for label in test_labels_npy:
			test_labels.append(int(label[3]))
		
		np.save('y_train.npy', train_labels)
		np.save('y_test.npy', test_labels)










if __name__ == '__main__':
	preprocess = Preprocess('Images/', 'Sequence_Labels/OPR/')
	preprocess.get_labels_from_files()