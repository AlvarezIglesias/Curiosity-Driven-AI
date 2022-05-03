import retro
import gym
import os
import time
import retro.enums


import numpy as np
from stable_baselines.common.atari_wrappers import wrap_deepmind
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecNormalize
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from PIL import Image
import imagehash

from sortedcontainers import SortedSet
from Discretizer import Discretizer

import time
time.clock = time.time


def create_q_model(num_actions):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(144, 160, 3))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(256, activation="relu")(layer4)
    #layer6 = layers.lstm()
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

def main():
    # Configuration paramaters for the whole setup
    tf.enable_eager_execution()

    seed = 42
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
            epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 16  # Size of batch taken from replay buffer
    max_steps_per_episode = 10000

    #
    env = retro.make("MarioBrosLand-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE,
                     use_restricted_actions=retro.Actions.ALL)
    #unlike in pokemon, you might need to jump and move at the same time
    env = Discretizer(env)
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    #env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(seed)

    num_actions = 8 #7 in pokemon

    # The first model makes the predictions for Q-values which are used to
    # make a action.
    model = create_q_model(num_actions)
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = create_q_model(num_actions)

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = keras.optimizers.Adam(learning_rate=0.1, clipnorm=1.0)
    #previous learning rate = 0.00025
    # Experience replay buffers
    action_history = []
    state_history = []
    state_hash_history = SortedSet()
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 500
    # Number of frames for exploration
    epsilon_greedy_frames = 10000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 1000
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    render_every = 1
    should_save = False
    debug_reward = False
    while True:  # Run until solved
        state = np.array(env.reset())
        episode_reward = 0
        state_hash_history = SortedSet()
        state_hash_history.add("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
        for timestep in range(1, max_steps_per_episode):
        #while True:

            if(timestep % render_every == 0): env.render();  # Adding this line would show the attempts
            # of the agent in a pop up window.
            #print("frame_count ", frame_count, "reward ", episode_reward)
            frame_count += 1

            # Use epsilon-greedy for exploration
            if  frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
                #print("Random action was taken")
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(tf.cast(state_tensor, tf.float32), training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)



            screen = Image.fromarray(state)
            state_hash = imagehash.whash(screen, hash_size=16)
            #imagehash.whash()

            ii = state_hash_history.bisect_left(str(state_hash))
            closest = state_hash_history[ii]#not really the closest always but anyways
            if closest == str(state_hash):
                reward = 0
            else:
                reward = abs(imagehash.old_hex_to_hash(closest, hash_size=16) - state_hash)/len(state_hash.hash)**2
                state_hash_history.add(str(state_hash))

            if debug_reward: print(reward)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                #indices = np.random.choice(range(len(done_history)), size=batch_size)
                indices = - np.arange(batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )
                #print(state_sample.shape)
                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(tf.cast(state_sample, tf.float32))

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))
                #if(should_save):
                model_target.save("saved_model")

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1
        print("starting episode", episode_count)
        #if running_reward > 40:  # Condition to consider the task solved
        #    print("Solved at episode {}!".format(episode_count))
        #    break


if __name__ == "__main__":
        main()

