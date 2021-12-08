import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.initializers
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.python.framework import dtypes

VGG_Feature_Shape = (7, 7, 512)
version = "25-1-SCE_GTNet"
D: int = 512

cce = CategoricalCrossentropy()

def symmetric_cross_entropy( alpha, beta ):
	"""
	modified version of @https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels/blob/master/loss.py
	:param alpha:
	:param beta:
	:return:
	"""
	
	def loss( y_true, y_pred ):
		return alpha * cce( y_true, y_pred ) + beta * cce( y_pred, y_true )
	
	return loss

def glove_embedding():
	try:
		embedding_matrix = np.load( "savedModels/gloveEmb.npz" )[ "arr_0" ]
		num_tokens, embedding_dim = embedding_matrix.shape
	except FileNotFoundError:
		
		path_to_glove_file = "savedModels/glove.6B.100d.txt"
		
		embeddings_index = { }
		with open( path_to_glove_file, encoding='utf-8' ) as f:
			for line in f:
				word, coefs = line.split( maxsplit=1 )
				coefs = np.fromstring( coefs, "f", sep=" " )
				embeddings_index[ word ] = coefs
		
		print( "Found %s word vectors." % len( embeddings_index ) )
		
		from dataset_code.dataset_utils import unique_names
		
		voc = unique_names()
		word_index = dict( zip( voc, range( len( voc ) ) ) )
		
		num_tokens = len( voc )
		embedding_dim = 100
		hits = 0
		misses = 0
		# Prepare embedding matrix
		embedding_matrix = np.zeros( (num_tokens, embedding_dim) )
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get( word.split( "_" )[ -1 ] )
			if embedding_vector is not None:
				# Words not found in embedding index will be all-zeros.
				# This includes the representation for "padding" and "OOV"
				embedding_matrix[ i ] = embedding_vector
				hits += 1
			else:
				misses += 1
		print( "Converted %d words (%d misses)" % (hits, misses) )
		
		np.savez_compressed( "savedModels/gloveEmb.npz", embedding_matrix )
	
	finally:
		return num_tokens, embedding_dim, embedding_matrix

def TX_Module( F, fGQ ):
	F_K = layers.Conv2D( filters=D, kernel_size=(1, 1), activation="linear" )( F )
	F_V = layers.Conv2D( filters=D, kernel_size=(1, 1), activation="linear" )( F )
	
	fGQ = K.reshape( fGQ, (-1, 1, D) )
	F_K = K.reshape( F_K, (-1, 49, D) )
	A = tf.nn.softmax( tf.matmul( fGQ, F_K, transpose_b=True ) / tf.sqrt( K.cast_to_floatx( D ) ) )
	
	A = K.reshape( A, (-1, 7, 7, 1) )
	# F_V = K.reshape( F_V, (-1, 7, 7, D) )
	fc = tf.math.multiply( A, F_V )
	fc = tf.reduce_sum( fc, axis=[ 1, 2 ] )
	fGQ = K.reshape( fGQ, (-1, D) )
	fc = layers.LayerNormalization()( layers.Add()( [ fc, fGQ ] ) )
	fc = layers.LayerNormalization()( layers.Add()( [ fc, layers.Dense( D, activation='relu' )( fc ) ] ) )
	return fc

def hoi_GTN_Model():
	# input1
	input1 = layers.Input( VGG_Feature_Shape, name="input1_imageF" )
	
	# input2
	input2 = layers.Input( shape=(80,), name="input2_obj1H" )
	obj_num = layers.Lambda( lambda x: tf.argmax( x, axis=1, output_type=dtypes.int32, ) )( input2 )
	hum_num = tf.fill( tf.shape( obj_num ), 15 )  # unique_names().index("person") ==15
	
	num_tokens, embedding_dim, embedding_matrix = glove_embedding()
	embeddingLayer = layers.Embedding( num_tokens, embedding_dim, embeddings_initializer=tensorflow.keras.initializers.Constant( embedding_matrix ),
	                                   input_length=1, trainable=False )
	
	hum_emb = embeddingLayer( hum_num )
	hum_emb = layers.LayerNormalization()( hum_emb )
	
	obj_emb = embeddingLayer( obj_num )
	obj_emb = layers.LayerNormalization()( obj_emb )
	
	f_W = layers.Concatenate()( [ hum_emb, obj_emb ] )
	f_W = layers.Dense( D, activation='relu' )( f_W )
	f_W = layers.LayerNormalization( name="f_W" )( f_W )
	
	# input4
	input4 = layers.Input( (64, 64, 2), name="input4_mask" )
	mask = layers.Conv2D( filters=32, kernel_size=5, activation="tanh" )( input4 )
	mask = layers.AveragePooling2D()( mask )
	mask = layers.Conv2D( filters=64, kernel_size=5, activation="tanh" )( mask )
	mask = layers.AveragePooling2D()( mask )
	mask = layers.Flatten()( mask )
	f_S = layers.Dense( D * 2, activation='relu' )( mask )
	f_S = layers.Dense( D, activation='relu' )( f_S )
	f_S = layers.LayerNormalization( name="f_S" )( f_S )
	
	# input6
	input6 = layers.Input( shape=(2, 4), name="input6_roiBoxes" )
	
	f_H = tf.image.crop_and_resize( input1, input6[ :, 0 ], tf.range( 0, tf.shape( input1 )[ 0 ], dtype=tf.int32 ), np.array( [ 7, 7 ] ) )
	f_O = tf.image.crop_and_resize( input1, input6[ :, 1 ], tf.range( 0, tf.shape( input1 )[ 0 ], dtype=tf.int32 ), np.array( [ 7, 7 ] ) )
	
	# todo: f_H and f_O  are missing residual blocks
	
	f_G = layers.GlobalAvgPool2D( name="f_G" )( input1 )
	f_H = layers.GlobalAvgPool2D( name="f_H" )( f_H )
	f_O = layers.GlobalAvgPool2D( name="f_O" )( f_O )
	
	f_B = layers.Concatenate()( [ f_G, f_H, f_O ] )
	f_B = layers.Dense( D * 2, activation='relu' )( f_B )
	f_B = layers.Dense( units=D, activation='relu' )( f_B )
	f_B = layers.LayerNormalization( name="f_B" )( f_B )
	
	# F_Ct
	FC_t = layers.Concatenate()( [ f_H, f_O ] )
	FC_t = layers.Dense( D * 2, activation='relu' )( FC_t )
	FC_t = layers.Dense( D, activation='relu' )( FC_t )
	f_Q = layers.LayerNormalization( name="f_Q" )( FC_t )
	
	f_GQ = layers.Multiply()( [ f_Q, f_S, f_W ] )
	
	f_C = TX_Module( input1, f_GQ )
	
	f_BR = layers.Multiply()( [ f_B, f_S, f_W ] )
	
	p_I = tf.nn.sigmoid( layers.Dense( D, activation='relu' )( layers.Concatenate()( [ f_B, f_BR, f_C ] ) ) )
	b_I = tf.nn.sigmoid( layers.Dense( D, activation='relu' )( layers.Concatenate()( [ f_B, f_BR ] ) ) )
	
	pHOI = layers.Multiply()( [ p_I, b_I ] )
	pHOI = layers.Dense( 117 )( pHOI )
	pHOI = layers.Activation( activation='softmax', name="output0_Class" )( pHOI )
	
	model = Model( inputs=[ input1, input6, input2, input4 ], outputs=pHOI )
	model.compile( loss=symmetric_cross_entropy( 0.5, 0.5 ),
	               optimizer='Adam',
	               metrics=AUC( name="mAP", curve="PR", thresholds=np.linspace( 0.5, 0.95, 10 ) ) )
	return model
