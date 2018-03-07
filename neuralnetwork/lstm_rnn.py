"""To generate a snippet for given input file name
   Documentation:
   		1.Dataset Preparation
   		2.convert the file into lower case
   		3.Create unique set of characters that contains in the document
   		4.Prepare the Training dataset with input sequence (Breakup the text with fixed length)
   		5.Prepare the LSTM network
   		6.Convert the document  into One Hot encoded
   		7.change  the output data into categorical"""



import numpy as np 
import pandas as pd 
import tensorflow as tf
import statsmodels
import math

list_already_encoded = [] #to store encoded list

def already_variable_is_encoded(output):
   status = None
   for encoded_num,count in list_already_encoded:
      if output == encoded_num:
         status = True
         break
      else:
         status = False

   return status #return the status

def do_hot_encode(data_Y):
   count = 0 #to keep track of count
   output_Y = [] #list to return
   for output in data_Y:
      if list_already_encoded == ' ':
         list_already_encoded.append((output,count))
         output_Y.append([count])
         count += 1 #increment the count
      else:
         #check if the output is already present in the list
         if(already_variable_is_encoded(output)):
            for num,count in list_already_encoded:
               if num == output:
                  output_Y.append([count])
         else:
            list_already_encoded.append((output,count))
            output_Y.append([count])
            count += 1

   return output_Y

#file to genearate a snippet
file_to_read = '/home/madhi/Documents/python programs/neuralnetworks/fp/Reuters21578-Apte-115Cat/training/cocoa/0000000'
#open as file
with open(file_to_read,'r')  as fp:
	file_text = fp.read()
	#convert  into lower
	file_text = file_text.lower() #convert the file into lower case

#create a set of unique characters
chars = sorted(list(set(file_text))) #total 43 characters
char_to_int = dict((c,i) for i,c in enumerate(chars))  #total chars 43 with their frequency in their document

#create a Training dataset
#the total length of characters in the file is 2898


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

#reshape the training data into Single Dimensional vector
X = np.reshape(data_X,(n_patterns,sequence_length,1)) # the matrix is in shape(2798,100,1)
#normalize the data
X_norm = X / int(n_vocab) #input data 


#one hot_encode the following
Y_norm = do_hot_encode(data_Y) #highest value is 14


#append 1 at the position of one hot_encoded variable
Y_output = [] #list to store the output vector
for encode in Y_norm:
   Y = [0]*15 #now we need to create a output vector
   pos = encode[0]
   Y[pos] = 1
   #append
   Y_output.append(Y)

#Y_output list has shape(?,15) <? number of output data>

hidden_nodes = 105
training_data = tf.placeholder(tf.float32,[None,100,1]) #<batch_size,seq_length,dimension>
training_output = tf.placeholder(tf.float32,[None,15]) #<batch_size,outputcell>

cell = tf.nn.rnn_cell.LSTMCell(hidden_nodes)
initial_state = cell.zero_state(X_norm.shape[1], tf.float32) #creating a tensor with initial_state with zeros

#the shape of the tensor will be (100,105) dtype =  float32
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell,training_data,
                                             initial_state=initial_state,dtype=tf.float32) 


#get the shape of the rnn_outputs
#transpose the matrix
rnn_out = tf.transpose(rnn_outputs,[1,0,2]) #transpose the matrix into 2 dimensions
last = tf.gather(rnn_out,int(rnn_out.get_shape()[0])-1) #value of output to the last tensor

print("Shape of the shape_rnn_output Matrix is {}".format(last))

#assigning weights and biases
bias =  tf.Variable(tf.constant(0.1,shape=[training_output.get_shape()[1]]))
weight =  tf.Variable(tf.truncated_normal([hidden_nodes,int(training_output.get_shape()[1])])) #tensor shape(105,15)

#prediction
prediction = tf.nn.softmax(tf.matmul(last,weight)+bias) # shape(100,15)
cross_entropy = -tf.reduce_sum(training_output * tf.clip_by_value(prediction,1e-10,1.0)) #calculate the cross_entropy

#optimizer
optimizer = tf.train.AdamOptimizer()
minimizer = optimizer.minimize(cross_entropy)

#initialize the graph and start the session
run_init_op = tf.global_variables_initializer()
sess = tf.Session() #start the session
sess.run(run_init_op) #run the Session

batch_size = 100 
no_of_batches = int((len(X_norm)) / batch_size) #run for 27 batches
epochs = 500 # number of epochs

#calculating mistakes and error_rates
mistakes = tf.not_equal(tf.argmax(training_output,1),tf.argmax(prediction,1))
error_rate = tf.reduce_mean(tf.cast(mistakes,tf.float32))


#run the loop
#the model runs for total 500 iterations
#run 500 iterations for total_batches

for i in range(epochs):
   ptr = 0
   for j in range(no_of_batches):
      inp,out = X_norm[ptr:ptr+batch_size],Y_output[ptr:ptr+batch_size]
      ptr += batch_size
      sess.run(minimizer,{training_data:inp,training_output:out})
   #print
   print("Epochs {} is processed".format(str(i)))
   incorrect = sess.run(error_rate,{training_data:inp,training_output:out})
   print('Epoch {0} error {1} %'.format(i + 1, 100 * error_rate))


#save the tensorflow model
save = saver.save(sess,'text_generate_trained_model.ckpt')
#result
result = rnn_outputs.evaluate(input_fn=training_data)
print("The Model result:")
print(result)
#close the session
sess.close() 