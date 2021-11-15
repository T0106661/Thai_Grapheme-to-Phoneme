from tensorflow.keras.models import load_model

import numpy as np

def load_data(filename):
	tups = []
	with open(filename, 'r') as f:
		for line in f:
			x, y = line.rstrip().split('\t')
			tups.append((x, y))
	return tups

def create_data(tups, x_idxs, y_idxs):
	x_maxlen  = max(len(x) for (x, y) in tups)
	ys_maxlen = max(len(y.split(' ')) for (x, y) in tups)
	
	x_data  = np.zeros((len(tups), x_maxlen),  dtype='int32')
	ys_data = np.zeros((len(tups), ys_maxlen), dtype='int32')
	yc_data = np.zeros((len(tups), ys_maxlen, 7), dtype='int32')
	for i, (x, y) in enumerate(tups):
		for j, c in enumerate(x):
			x_data [i, j] = x_idxs[c]
		for j, s in enumerate(y.split(' ')):
			ys_data[i, j] = y_idxs[f's_{s}']
			for k, c in enumerate(s):
				yc_data[i, j, k] = y_idxs[f'c_{c}']
	
	return x_data, ys_data, yc_data

if __name__ == '__main__':
	import argparse
	import pickle
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--input',  required=True)
	parser.add_argument('--model',  required=True)
	parser.add_argument('--output', required=True)
	args   = parser.parse_args()
	
	with open('ids.pkl', 'rb') as f:
		x_idxs, y_idxs = pickle.load(f)
	
	tups   = load_data(args.input)
	model  = load_model(args.model)
	x_data, ys_data, yc_data = create_data(tups, x_idxs, y_idxs)
	z_data = model.predict([x_data, ys_data, yc_data])
	with open(args.output, 'w') as f:
		for i, (x, y) in enumerate(tups):
			print(f'{x}\t{y}\t{z_data[i, 0]}', file=f)

