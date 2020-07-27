import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf
import keras
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop, Adam, SGD
# from tensorflow import keras
import random
import time 
from datetime import datetime
import os
import json

config = None
with open('config.json') as f:
  config = json.load(f)
  print(config)
#   print(config["env_type"])
#   config = json.dumps(config_dic)
#   print(config.env_type)


""" start_epsilon = 1
end_epsilon = 0.01
epsilone = start_epsilon
decay_rate = 1.001
learning_rate = 0.0025
train_dir = "Check_point"
# restore_path = "Check_point/CartPole-v0_20200317112349_reward_54.0_frames_15445.h5"
# restore_path = "Best_Weights/Pong-ram-v4_20200419133544_reward_7.0_frames_11560138.h5"
# restore_path = "Check_point/Breakout-ram-v4_20200421063249_reward_97.0_frames_9889405.h5"
restore_path = "Check_point/Breakout-ram-v4_20200718083145_reward_106.0_frames_820356.h5"
# restore_path = "CheckPoints/Breakout-ram-v4_20200421065718_reward_63.0_frames_10310154.h5"
after_frame_learning_start = 20000
update_model_frq = 4
update_target_model = 8000
training_episod = 100000
current_frame = 0
# current_frame = 104633
current_episod = 0
render = True
resume = True
DEVLOPER_TESTING = False
training = True
testing = False
save_model = True
# env_type = "3D"  #2D
env_type = "2D"

# render_mode = "machine" #"human"
render_mode = "human"

batch_size = 32
memory = []
memory_max_size = 400000
 """
start_epsilon = 1
end_epsilon = 0.01
epsilone = start_epsilon
decay_rate = config["decay_rate"]
learning_rate = config["learning_rate"]
train_dir = "Check_point"
# restore_path = "Check_point/CartPole-v0_20200317112349_reward_54.0_frames_15445.h5"
# restore_path = "Best_Weights/Pong-ram-v4_20200419133544_reward_7.0_frames_11560138.h5"
# restore_path = "Check_point/Breakout-ram-v4_20200421063249_reward_97.0_frames_9889405.h5"
restore_path = config["check_point"]
# restore_path = "CheckPoints/Breakout-ram-v4_20200421065718_reward_63.0_frames_10310154.h5"
after_frame_learning_start = config["learning_start_after_frames"]
update_model_frq = config["update_main_model"]
update_target_model = config["update_target_model"]
training_episod = config["num_of_game_play"]
current_frame = 0
# current_frame = 104633
current_episod = 0
render = config["render"]
resume = config["training"]["resume"]
DEVLOPER_TESTING = False
training = config["training"]["training"]
save_model = config["training"]["save_model"]
testing = config["testing"]["testing"]
# env_type = "3D"  #2D
env_type = config["env_type"]

render_mode = "machine" #"human"
# render_mode = "human"

batch_size = config["training"]["batch_size"]
memory = []
memory_max_size = config["training"]["memory_max_size"]

ACTION_SPACE = 3
OBSERV_SPACE = (84, 84, 4)




def pre_process(state):
    if env_type == "3D":
        return np.uint8(resize(rgb2gray(state), (84, 84), mode="constant")) 
    else:
        return [state]
    # print(state.shape)

def add_memory(history , action , reward , next_history , is_done):
    if current_frame < memory_max_size:
        memory.append((history, action, reward, next_history, is_done))
    else:
      if not resume:
        memory[current_frame % memory_max_size] = (history, action, reward, next_history, is_done)
      else:
        memory[current_frame % (memory_max_size - after_frame_learning_start)] = (history, action, reward, next_history, is_done)
def agent_model():
    frame_input  = layers.Input(shape=OBSERV_SPACE , name="frame")
    action_input = layers.Input(shape=(ACTION_SPACE , ) , name="action")

    if env_type == "3D":
        normalized = layers.Lambda(lambda x: x , name='normalization')(frame_input)
        conv_1 = layers.Conv2D(16 , (8 , 8) , strides=(4 , 4) , activation="relu" , kernel_initializer='random_uniform' ,bias_initializer='random_uniform'  )(normalized)
        conv_2 = layers.Conv2D(32 , (4 , 4) , strides=(2 , 2) , activation="relu" , kernel_initializer='random_uniform' ,bias_initializer='random_uniform'  )(conv_1)
        flatten = layers.Flatten()(conv_2)
        dense_1 = layers.Dense(256, activation="relu" ,kernel_initializer='random_uniform' ,bias_initializer='random_uniform'  )(flatten)
    elif env_type == "2D":
        flatten = layers.Flatten()(frame_input)
        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(flatten)
        dense_1_c = layers.Dense(256, activation="relu" , kernel_initializer='random_uniform' ,bias_initializer='random_uniform' )(normalized)
        dense_1 = layers.Dense(256, activation="relu" , kernel_initializer='random_uniform' ,bias_initializer='random_uniform' )(dense_1_c)
    else:
        print("Exception please set enviournment type like '3D' , '2D' ")
        exit()
    
    out_1 = layers.Dense(ACTION_SPACE)(dense_1)

    mult_output = layers.Multiply(name = "Q_val")([out_1 , action_input])

    model = Model(inputs=[frame_input , action_input] , outputs= mult_output)

    optimizer = RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01)
    # optimizer = Adam(learning_rate=learning_rate)
    # optimizer = SGD(learning_rate=learning_rate)

    model.compile(optimizer , loss="mse")

    return model


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def batch_fit(model ,target_model):
    mini_batch = random.sample(memory,batch_size)
    history = np.zeros((batch_size , OBSERV_SPACE[0] , OBSERV_SPACE[1] , 4))
    next_history = np.zeros((batch_size, OBSERV_SPACE[0], OBSERV_SPACE[1], 4))
    actions , rewards , is_dones =  [] , [] , []
    for idx , val in enumerate(mini_batch):
        history[idx] = val[0]
        actions.append(val[1])
        rewards.append(val[2])
        next_history[idx] = val[3]
        is_dones.append(val[4])

    next_q_val = target_model.predict([next_history , np.ones((batch_size , ACTION_SPACE))])    
    # q_val_ = model.predict([history , np.ones((batch_size , ACTION_SPACE))])    
    target = np.zeros((batch_size , ))
    # target = q_val_
    for i in range(batch_size):
        if is_dones[i]:
            target[i] = rewards[i]
        else:
            target[i] = rewards[i] + 0.99 * np.amax(next_q_val[i])
    
    # action_one_hot = np.ones((batch_size, ACTION_SPACE))
    action_one_hot = get_one_hot(actions , ACTION_SPACE)

    target_one_hot = action_one_hot * target[: , None]

    # print(np.sum((target_one_hot - next_q_val)**2))

    # print(target_one_hot)
    # exit()
    
    h = model.fit([history , action_one_hot] , target_one_hot , epochs=1 , batch_size=batch_size , verbose=0)
    return h.history['loss'][0]
    # return np.sum((target_one_hot - next_q_val)**2)
    # return


def deep_copy_model(model):
    copy_model = keras.models.clone_model(model)
    copy_model.set_weights(model.get_weights())
    return copy_model

def train():
    # GAME = "Pong-v0"
    global ACTION_SPACE , OBSERV_SPACE
    # GAME = "Boxing-v0"
    # GAME = "KungFuMaster-v4"
    # GAME = "QbertDeterministic-v4"
    # GAME = "PongDeterministic-v4"
    # GAME = "FrostbiteDeterministic-v4"
    # GAME = "KungFuMasterDeterministic-v4"
    # GAME = "Frostbite-ram-v0"
    # GAME = "CrazyClimber-ram-v0"
    # GAME = "KungFuMaster-ram-v0"
    # GAME = "Qbert-ram-v0"
    # GAME = "Boxing-ram-v0"
    # GAME = "Breakout-ram-v0"
    # GAME = "Breakout-ram-v4"
    # GAME = "Pong-ram-v4"
    # GAME = "CartPole-v1"
    # GAME = "LunarLander-v2"
    # GAME = "Carnival-v4"
    # GAME = "SpaceInvaders-ram-v0"
    # GAME = "Riverraid-ram-v0"
    # GAME = "Assault-ram-v0"

    GAME = config["game_name"]
    env = gym.make(GAME)
    
    if render_mode == "human":
        env.render(mode='human')

    global current_frame
    global epsilone
    global end_epsilon
    print(env.action_space)
    ACTION_SPACE = env.action_space.n
    # ACTION_SPACE = 3
    if env_type == "3D":
        OBSERV_SPACE = (84 , 84 , 4)
    else:
        OBSERV_SPACE = (1, env.observation_space.shape[0], 4)

    if not resume:
        model = agent_model()
        target_model = deep_copy_model(model)
    else:
        model = keras.models.load_model(restore_path)
        target_model = deep_copy_model(model)
        end_epsilon = 0
        epsilone = end_epsilon
        # global current_frame
        # print(after_frame_learning_start)
        current_frame = after_frame_learning_start
    
    observation = env.reset()

    for _ in range(0, 200):
        observation,_ ,_ , _ = env.step(random.randrange(ACTION_SPACE))
    # print(observation[0][:], observation[1][:] , observation[2][:] , observation[3][:])
    # exit()
    observation = pre_process(observation)
    history = np.stack((observation, observation, observation, observation) , axis=2)
    history = np.reshape([history], (1, OBSERV_SPACE[0], OBSERV_SPACE[1], 4))
    print(current_frame , "Frame")
    for i_episode in range(training_episod):
        loss = 0.0
        total_reward = 0
        is_done = False
        observation = env.reset()
        checked_frame = current_frame
        temp_count = 0
        while not is_done:
            if render:
                if render_mode == "human":
                    env.render(mode='human')
                else:
                    env.render()

            if (current_frame > after_frame_learning_start and epsilone < random.random()):
                q_val = model.predict([history , np.ones(ACTION_SPACE).reshape(1 , ACTION_SPACE)])
                action = np.argmax(q_val)
            else:
                # action = env.action_space.sample()
                action = random.randrange(ACTION_SPACE)
                # print("okay" , action)
            if current_frame > after_frame_learning_start:
                if DEVLOPER_TESTING:
                    epsilone = 0
                else:
                    # global end_epsilon
                    epsilone = end_epsilon + (start_epsilon - end_epsilon)* (decay_rate)**(-current_frame)
            # print("hi" , action)
            real_action = action
            observation, reward, is_done, info = env.step(real_action)
            # print("+++++++++++++++++++++++++++++++++++")
            # # print(observation.shape)
            # # print(observation[:][:])
            # # exit()
            observation_crop = pre_process(observation)
            total_reward += reward
            # print(observation.shape)
            # print(history[:, :, :, :3].shape)
            next_history = np.append(np.reshape([observation_crop] , (1,OBSERV_SPACE[0],OBSERV_SPACE[1],1)) , history[:, : , : , :3] , axis=3)
            # print(next_history , next_history.shape)
            add_memory(history , action , reward , next_history , is_done)

            # print(history == next_history)
            history = next_history.copy()

            if(current_frame > after_frame_learning_start and DEVLOPER_TESTING == False and i_episode != 0):
                if(current_frame % update_model_frq == 0):
                    loss += batch_fit(model , target_model)/32
                # print(loss)
                if current_frame % update_target_model == 0:
                    # print("update target model")
                    target_model.set_weights(model.get_weights())
                
            # exit()
            current_frame += 1
        # print(loss)
        print("Episod: {} , Epsilone {}, reward: {} , total_frame: {} , loss: {}".format(i_episode ,epsilone , total_reward ,current_frame , loss/(current_frame-checked_frame) ))

        if save_model:
            if((i_episode % 10 == 0 or i_episode + 1 == training_episod) and epsilone != 1):
                ct_time = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                file_name = "{}_{}_reward_{}_frames_{}.h5".format(GAME , ct_time , total_reward , current_frame)
                model_path = os.path.join(train_dir , file_name)
                model.save(model_path)
def test():
    model = keras.models.load_model(restore_path)
    target_model = deep_copy_model(model)
    global ACTION_SPACE, OBSERV_SPACE
    # GAME = "CartPole-v0"
    # GAME = "Breakout-ram-v4"
    GAME = config["game_name"]
    env = gym.make(GAME)
    env.mode = 'human'
    # env._configure(batch_mode=False)

    ACTION_SPACE = env.action_space.n
    if env_type == "3D":
        OBSERV_SPACE = (84 , 84 , 4)
    else:
        OBSERV_SPACE = (1, env.observation_space.shape[0], 4)

    # ACTION_SPACE = 3
    # OBSERV_SPACE = (1, env.observation_space.shape[0], 4)

    observation = env.reset()

    for _ in range(0, 20):
        env.render()
        observation, _, _, _ = env.step(random.randrange(ACTION_SPACE))
    observation = pre_process(observation)
    history = np.stack((observation, observation, observation, observation), axis=2)
    history = np.reshape([history], (1, OBSERV_SPACE[0], OBSERV_SPACE[1], 4))
    print("okay")
    for i_episode in range(training_episod):
        global current_frame
        # loss = 0.0
        total_reward = 0
        is_done = False
        observation = env.reset()
        print("okay2")
        while not is_done:
            # if render:
            env.render()
            # print("okay3")
            q_val = target_model.predict([history, np.ones(ACTION_SPACE).reshape(1, ACTION_SPACE)])
            action = np.argmax(q_val)
            # print(action)
            real_action = action

            observation, reward, is_done, info = env.step(real_action)
            observation = pre_process(observation)
            total_reward += reward
            # print(observation.shape)
            # print(history[:, :, :, :3].shape)
            next_history = np.append(np.reshape([observation], (1, OBSERV_SPACE[0], OBSERV_SPACE[1], 1)), history[:, :, :, :3], axis=3)
            # print(next_history , next_history.shape)
            # exit()
            history = next_history
            current_frame += 1
            time.sleep(config["testing"]["sleep_btwn_frame"])
        print("Total_reward {}".format(total_reward))
    env.close()
if training:
    train()
elif testing:
    test()
""" if not resume:
    model = agent_model()
    target_model = deep_copy_model(model)



for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print(pre_process(observation))
        observation = pre_process(observation)
        history = np.stack((observation, observation, observation, observation ), axis=2)
        history = np.reshape([history] , (1 , 84, 84, 4))
        print(history , history.shape)
        print(model.predict( [history , np.ones(3).reshape((1, 3)) ]) )
        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
        break
env.close() """
