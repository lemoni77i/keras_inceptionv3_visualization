import argparse

import lucid.optvis.objectives as objectives
import lucid.optvis.render as render
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from lucid.misc.io import show
from lucid.modelzoo.vision_base import Model
from vis.visualization import visualize_saliency


def get_strongest_filter(model, layer, img_, n):
	img = img_
	width = img.shape[0]
	height = img.shape[1]

	# calculate the activations:
	with tf.Graph().as_default(), tf.Session():
		t_input = tf.placeholder(tf.float32, [width, height, 3])
		T = render.import_model(model, t_input, t_input)
		activations = T(layer).eval({t_input: img})[0]

		# get n filters with highest activation
		MAX = activations.argmax(-1)
		MAX_1 = MAX.flatten()
		COUNTS = np.bincount(MAX_1)
		N_MAX_FILTERS = np.argsort(-COUNTS)[:n]

	return N_MAX_FILTERS


def render_vis(model, objective_f, file_name, filter_idx,
			   param_f=None, optimizer=None, transforms=None, thresholds=(512,),
			   verbose=True, relu_gradient_override=True, use_fixed_seed=False):
	with tf.Graph().as_default() as graph, tf.Session() as sess:

		if use_fixed_seed:  # does not mean results are reproducible, see Args doc
			tf.set_random_seed(0)

		T = render.make_vis_T(model, objective_f, param_f, optimizer, transforms,
							  relu_gradient_override)
		loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
		tf.global_variables_initializer().run()

		loss_p = 0
		try:
			for i in range(max(thresholds) + 1):
				loss_, _ = sess.run([loss, vis_op])
				if i in thresholds:
					vis = t_image.eval()
					loss_p = loss_
					plt.title("Filter {}, {:.5f}".format(filter_idx, loss_))
					plt.imsave(file_name, np.hstack(vis))
		except KeyboardInterrupt:
			vis = t_image.eval()
			show(np.hstack(vis))

		return loss_p


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="args parser")
	parser.add_argument("-modelzoo_file")
	parser.add_argument("-conv_layer_idx", type=int)
	parser.add_argument("-filter_idx", type=int)  # optional
	parser.add_argument("-num_of_filters", type=int)
	parser.add_argument("-input_image_path")
	args = parser.parse_args()

	class LucidInceptionV3(Model):
		# model_path = 'keras_inception_v3_frozen.pb.modelzoo'
		model_path = args.modelzoo_file
		image_shape = [299, 299, 3]
		image_value_range = (0, 1)
		input_name = 'input_1'

	keras_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3))
	inceptionv3 = LucidInceptionV3()
	inceptionv3.load_graphdef()

	# Image Processing
	img_path = args.input_image_path
	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = preprocess_input(x)
	y = image.img_to_array(img)
	y = np.expand_dims(y, axis=0)
	y = preprocess_input(y)

	layer_idx = args.conv_layer_idx
	for i in range(len(keras_model.layers)):
		if keras_model.layers[i].name == "conv2d_{}".format(layer_idx):
			keras_layer_idx = i
			break
	layer_name = "conv2d_{}/convolution".format(layer_idx)
	filter_idx = args.filter_idx
	n_of_filter = args.num_of_filters

	if filter_idx is None:
		filters = [int(idx) for idx in get_strongest_filter(inceptionv3, layer_name, x, n_of_filter)]
	else:
		filters = [filter_idx]

	for idx in filters:
		obj = objectives.channel(layer_name, idx)

		file_name = "Conv{}_{}".format(layer_idx, idx)
		img_file = file_name + ".png"

		render_vis(inceptionv3, obj, img_file, idx)
		saliency_img = visualize_saliency(keras_model, keras_layer_idx, filter_indices=idx, seed_input=y)
		plt.imshow(img)
		plt.imshow(saliency_img, alpha=.7)
		plt.axis('off')
		plt.savefig("{}_saliency.png".format(file_name))
