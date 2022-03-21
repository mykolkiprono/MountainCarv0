import gym 
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

EPISODES = 400


class DQNAagent:
	def __init__(self,state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size

		self.memory = deque(maxlen = 2000)
		self.gamma = 0.97
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.learning_rate = 0.01
		self.decay = 0.97
		self.model = self._build_model()
		self.target_model = self._build_model()
		self.update_model

	def _build_model(self):
		model = Sequential()

		model.add(Dense(64, input_dim = 2))
		model.add(Activation('relu'))

		model.add(Dense(64))
		model.add(Activation('relu'))

		model.add(Dense(self.action_size, activation = 'linear'))
		model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = self.learning_rate))
		return model 

	def update_model(self):
		self.target_model.set_weights(self.model.get_weights)


	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		minibatch = self.memory.random.sample(batch_size)

		for state, action, reward, next_state, done in minibatch:
			target = self.model.predict(state)

			if done:
				target[0][action] = reward
			q_future = self.model.predict(state)
			target[0][action] = reward + self.gamma * np.amax(q_future)
			self.model.fit(state, target, epochs = 1, verbose = 0)

			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay

	def save(self, name):
		self.model.save(name)           



if __name__ == '__main__':
	env = gym.make('MountainCar-v0')
	state_size = env.observation_space
	action_size = env.action_space.n 
	agent = DQNAagent(state_size, action_size)

	print(f"state_size {state_size}")
	print(f"action_size {action_size}")

	done =False 
	batch_size = 128 

	scores = 0 

	for e in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		flag = 0 
		
		for time in range(200):
			env.render()

			action = agent.act(state)
			next_state, reward, done, info = env.step(action)

			if next_state[1] > state[0][1] and next_state[1]>0 and state[0][1]>0:
				reward+=15

			elif next_state[1] < state[0][1] and next_state[1]<=0 and state[0][1]<=0:
				reward +=15

			if done:
				reward = reward + 10000

			else:
				reward = reward - 10

			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			scores += reward
			if done:
				flag = 1
				agent.update_target_model()
				print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, scores, agent.epsilon))
				break
			if len(agent.memory) > batch_size:
				agent.replay(batch_size)
				if flag == 0:
					print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))

				if e % 100 == 0:
					print('saving the model')
					agent.save("mountain_car-dqn.h5")





