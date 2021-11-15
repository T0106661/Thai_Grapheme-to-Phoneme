from tensorflow.keras.models    import Model
from tensorflow.keras.layers    import Input, Embedding, GRU, Bidirectional, TimeDistributed, Dense, concatenate 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np

def load_data(filename):
	tups   = []
	x_chrs = set()
	y_syls = set()
	y_chrs = set()
	with open(filename, 'r') as f:
		for line in f:
			x, y, z = line.rstrip().split('\t')
			tups.append((x, y, float(z)))
			x_chrs |= set(x)
			for s in y.split(' '):
				y_syls.add(s)
				y_chrs |= set(s)
	
	x_chrs = [None] + sorted(x_chrs)
	x_idxs = {c: i for i, c in enumerate(x_chrs)}
	y_syls = [None] + sorted(y_syls)
	y_chrs = [None] + sorted(y_chrs)
	y_idxs = {None: 0}
	for i, s in enumerate(y_syls):
		if i != 0:
			y_idxs[f's_{s}'] = i
	for i, c in enumerate(y_chrs):
		if i != 0:
			y_idxs[f'c_{c}'] = i
	
	return tups, x_chrs, x_idxs, y_syls, y_chrs, y_idxs

def create_data(tups, x_idxs, y_idxs):
	x_maxlen  = max(len(x) for (x, y, z) in tups)
	ys_maxlen = max(len(y.split(' ')) for (x, y, z) in tups)
	yc_maxlen = max(len(s) for (x, y, z) in tups for s in y.split(' '))
	
	x_data  = np.zeros((len(tups), x_maxlen),  dtype='int32')
	ys_data = np.zeros((len(tups), ys_maxlen), dtype='int32')
	yc_data = np.zeros((len(tups), ys_maxlen, yc_maxlen), dtype='int32')
	z_data  = np.zeros((len(tups), 1), dtype='float32')
	for i, (x, y, z) in enumerate(tups):
		for j, c in enumerate(x):
			x_data [i, j] = x_idxs[c]
		for j, s in enumerate(y.split(' ')):
			ys_data[i, j] = y_idxs[f's_{s}']
			for k, c in enumerate(s):
				yc_data[i, j, k] = y_idxs[f'c_{c}']
		z_data[i, 0] = z
	
	return x_data, ys_data, yc_data, z_data

def create_model(x_chrs, y_syls, y_chrs, p=0.2):
	x_input  = Input(shape=(None,))
	ys_input = Input(shape=(None,))
	yc_input = Input(shape=(None, None))
	
	x  = x_input
	x  = Embedding(len(x_chrs), 32, mask_zero=True)(x)
	x  = Bidirectional(GRU(128, dropout=p, recurrent_dropout=p))(x)
	
	ys = ys_input
	ys = Embedding(len(y_syls), 64, mask_zero=True)(ys)
	
	yc = yc_input
	yc = Embedding(len(y_chrs), 16, mask_zero=True)(yc)
	yc = TimeDistributed(Bidirectional(GRU(32)))(yc)
	
	y  = concatenate([ys, yc])
	y  = Bidirectional(GRU(256, dropout=p, recurrent_dropout=p))(y)
	
	z  = concatenate([x, y])
	z  = Dense(96, activation='tanh')(z)
	z  = Dense(1)(z)
	
	model = Model([x_input, ys_input, yc_input], z)
	model.compile('adam', 'mae')
	return model

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', required=True)
	parser.add_argument('--model', required=True)
	args   = parser.parse_args()
	
	tups, x_chrs, x_idxs, y_syls, y_chrs, y_idxs = load_data(args.input)
	model = create_model(x_chrs, y_syls, y_chrs)
	x_data, ys_data, yc_data, z_data = create_data(tups, x_idxs, y_idxs)
	
	ckpt  = ModelCheckpoint(args.model, save_best_only=True)
	early = EarlyStopping(monitor='val_loss', patience=3)
	model.fit(
		[x_data, ys_data, yc_data], z_data, validation_split=0.1,
		batch_size=128, epochs=100,
		callbacks=[ckpt, early]
	)
