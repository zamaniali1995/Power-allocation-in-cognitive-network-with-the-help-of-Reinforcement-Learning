import tensorflow as tf
from tensorflow.keras import layers, Model
# from layers import Input, Dense
import numpy as np 
import random
from collections import deque 

GAMMA = 0.8  
OBSERVE = 300 
EXPLORE = 100000 
FINAL_EPSILON = 0.0 
INITIAL_EPSILON = 0.8 
REPLAY_MEMORY = 400 
BATCH_SIZE = 256 

class BrainDQN:
	def __init__(self, actions, PU_num, SU_num):

		self.replayMemory = []
		for _ in range(SU_num):
			self.replayMemory.append(deque())
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.recording = EXPLORE
		# self.sensor_dim = Sensor
		self.PU_num = PU_num
		self.SU_num = SU_num
		self.actions = actions
		self.hidden1 = 256
		self.hidden2 = 256
		self.hidden3 = 512
        
		self.createQNetwork()

	def createQNetwork(self):
		x_su = []
		self.stateInput = tf.placeholder("float64",[None, self.SU_num])
		x = tf.keras.layers.Dense(self.hidden1, activation='relu')(self.stateInput)
		x = tf.keras.layers.Dense(self.hidden2, activation='relu')(x)
		x = tf.keras.layers.Dense(self.hidden3, activation='relu')(x)
		
		for i in range(self.SU_num):
			x_su_1 = tf.keras.layers.Dense(512, activation='relu')(x)
			x_su.append(tf.keras.layers.Dense(512, activation='relu')(x_su_1))
		
		self.output = []
		for i in range(self.SU_num):
			self.output.append(tf.keras.layers.Dense(self.actions, activation='tanh', name='output_su_'+str(i))(x_su[i]))
		

		self.actionInput = tf.placeholder("float64",[None,self.actions])
		self.yInput = []
		
		self.cost = 0
		for i in range(self.SU_num):
			self.yInput.append(tf.placeholder("float64", [None]))  
			Q_action = tf.reduce_sum(tf.multiply(self.output[i], self.actionInput), reduction_indices = 1)
			self.cost += tf.reduce_mean(tf.square(self.yInput[i] - Q_action))
		
		self.trainStep = tf.train.AdamOptimizer(learning_rate=10**-5).minimize(self.cost)

		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())

	def trainQNetwork(self):
		y_batch = [[] for _ in range(self.SU_num)]
		for i in range(self.SU_num):
			minibatch = random.sample(self.replayMemory[i], BATCH_SIZE)
			state_batch = [data[0] for data in minibatch]
			action_batch = [data[1] for data in minibatch]
			reward_batch = [data[2] for data in minibatch]
			nextState_batch = [data[3] for data in minibatch]
			
			for j in range(0,BATCH_SIZE):
				QValue_batch = self.output[i].eval(feed_dict={self.stateInput:[nextState_batch[j]]})
				y_batch[i].append(reward_batch[j] + GAMMA * np.max(QValue_batch))

		feed_dictionary = {}
		for (yI, yB) in zip(self.yInput, y_batch):
			feed_dictionary[yI] = yB
		feed_dictionary[self.actionInput] = action_batch
		feed_dictionary[self.stateInput] = state_batch 
		_, self.loss = self.session.run([self.trainStep,self.cost],feed_dict=feed_dictionary)
		return self.loss

	def setPerception(self,nextObservation,action_index,reward):
		loss = 0
		newState = nextObservation
		for i in range(self.SU_num):
			action = np.zeros(self.actions)
			action[action_index[i]] = 1
			self.replayMemory[i].append((self.currentState, action, reward, newState))
			if len(self.replayMemory[i]) > REPLAY_MEMORY:
				self.replayMemory[i].popleft()
		if self.timeStep > OBSERVE:
            
			loss = self.trainQNetwork()

		self.currentState = newState
		self.timeStep += 1
		return loss
        

	def getAction(self):
		QValue = []
		for i in range(self.SU_num):
			QValue.append(self.output[i].eval(feed_dict= {self.stateInput:[self.currentState]}))

		action_index = []
		if random.random() <= self.epsilon:
			for i in range(self.SU_num):
				action_index.append(random.randrange(self.actions))
		else:
			for i in range(self.SU_num):
				action_index.append(np.argmax(QValue[i]))
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
			self.recording = self.recording-1

		return action_index, self.recording
    
	def getAction_test(self,observation):

		action_index = []
		for i in range(self.SU_num):
			QValue = self.output[i].eval(feed_dict= {self.stateInput:[self.currentState]})
			action_index.append(np.argmax(QValue))
		return action_index
    
	def setInitState(self, observation):
		self.currentState = observation
		
	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)
            
