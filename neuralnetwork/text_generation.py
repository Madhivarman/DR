import tensorflow as tf 
import numpy as np 

#load the file to create a Text 
file = '/home/madhi/Documents/python programs/neuralnetworks/fp/Reuters21578-Apte-115Cat/training/cocoa/0000000'

with open(file,'r')  as fp:
	file_text = fp.read()
	#convert  into lower
	file_text = file_text.lower() #convert the file into lower case

#create a set of unique characters
chars = sorted(list(set(file_text))) #total 43 characters
char_to_int = dict((c,i) for i,c in enumerate(chars))  #total chars 43 with their frequency in their document

#int_to_char change
int_to_char = dict((i,c) for i,c in enumerate(chars))

sequence_length = 100
n_vocab = len(chars) #length of the characters
n_chars = len(file_text) #length of the total words in a file
data_X = [] #dataset
data_Y = [] #output to the dataset

for i in range(0,n_chars - sequence_length,1):
	seq_in = file_text[i:i+sequence_length] #separate the text into slices
	seq_out = file_text[i+sequence_length] #output for the input

	data_X.append([char_to_int[char]  for char in seq_in])   #append to the training data
	data_Y.append([char_to_int[seq_out]]) #append output to the file

n_patterns = len(data_X) #2798

#now pick a random seed
start = np.random.randint(0,len(data_X)-1)
pattern = data_X[start]

print("Seed")
print("\"".join([int_to_char[value] for value in pattern]),"\"")

#start the session
#reset the defaule graph

tf.reset_default_graph()
#load the meta data graph
imported_meta = tf.train.import_meta_graph('text_generate_trained_model.ckpt.meta')

with tf.Session() as sess:
	imported_meta.restore(sess,tf.train.latest_checkpoint('./'))

	#accessing the default graph which we restored
	graph = tf.get_default_graph()

	#op that we can be processed to get the output
	#last is the tensor that is the prediction of the network
	y_pred = graph.get_tensor_by_name("prediction:0")
	#generate characters
	for i in range(500):
		x = np.reshape(pattern,(1,len(pattern),1))
		x = x / float(n_vocab)
		prediction = sess.run(y_pred,feed_dict=x)
		index = np.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		sys.stdout.write(result)
		patter.append(index)
		pattern = pattern[1:len(pattern)]

	print("\n Done...!")


sess.close()