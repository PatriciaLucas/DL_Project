import io
import base64
from IPython.display import HTML
import gym
import numpy as np
import cv2
import warnings

def play_video(filename, width=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    encoded = base64.b64encode(io.open(filename, 'r+b').read())
    video_width = 'width="' + str(width) + '"' if width is not None else ''
    embedded = HTML(data='''
        <video controls {0}>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>'''.format(video_width, encoded.decode('ascii')))

    return embedded


def preprocess_pong(image):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    I = image[35:195] # Crop
    I = I[::2, ::2, 0] # Downsample width and height by a factor of 2
    I[I == 144] = 0 # Remove background type 1
    I[I == 109] = 0 # Remove background type 2
    I[I != 0] = 1 # Set remaining elements (paddles, ball, etc.) to 1
    I = cv2.dilate(I, np.ones((3, 3), np.uint8), iterations=1)
    I = I[::2, ::2, np.newaxis]
    return I.astype(np.float)


def pong_change(prev, curr):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    prev = preprocess_pong(prev)
    curr = preprocess_pong(curr)
    I = prev - curr
    # I = (I - I.min()) / (I.max() - I.min() + 1e-10)
    return I


class Memory:
  def __init__(self): 
      self.clear()

  # Resets/restarts the memory buffer
  def clear(self): 
      self.observations = []
      self.actions = []
      self.rewards = []

  # Add observations, actions, rewards to memory
  def add_to_memory(self, new_observation, new_action, new_reward): 
      self.observations.append(new_observation)
      self.actions.append(new_action)
      self.rewards.append(new_reward)

    
def aggregate_memories(memories):
  warnings.filterwarnings("ignore", category=DeprecationWarning) 
  batch_memory = Memory()
  for memory in memories:
    for step in zip(memory.observations, memory.actions, memory.rewards):
      batch_memory.add_to_memory(*step)
  return batch_memory


def parallelized_collect_rollout(batch_size, envs, model, choose_action):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    assert len(envs) == batch_size
    memories = [Memory() for _ in range(batch_size)]
    next_observations = [single_env.reset() for single_env in envs]
    previous_frames = [obs for obs in next_observations]
    done = [False] * batch_size
    rewards = [0] * batch_size
    while True:
        current_frames = [obs for obs in next_observations]
        diff_frames = [pong_change(prev, curr) for (prev, curr) in zip(previous_frames, current_frames)]
        diff_frames_not_done = [diff_frames[b] for b in range(batch_size) if not done[b]]
        actions_not_done = choose_action(model, np.array(diff_frames_not_done), single=False)
        actions = [None] * batch_size
        ind_not_done = 0
        for b in range(batch_size):
            if not done[b]:
                actions[b] = actions_not_done[ind_not_done]
                ind_not_done += 1

        for b in range(batch_size):
            if done[b]:
                continue
            next_observations[b], rewards[b], done[b], info = envs[b].step(actions[b])
            previous_frames[b] = current_frames[b]
            memories[b].add_to_memory(diff_frames[b], actions[b], rewards[b])

        if all(done):
            break

    return memories


def save_video_of_model(model, env_name, suffix=""):
    import skvideo.io
    from pyvirtualdisplay import Display
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    display = Display(visible=0, size=(400, 300))
    display.start()

    env = gym.make(env_name)
    obs = env.reset()
    prev_obs = obs

    filename = env_name + suffix + ".mp4"
    output_video = skvideo.io.FFmpegWriter(filename)

    counter = 0
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        output_video.writeFrame(frame)

        if "CartPole" in env_name:
            input_obs = obs
        elif "Pong" in env_name:
            input_obs = pong_change(prev_obs, obs)
        else:
            raise ValueError(f"Unknown env for saving: {env_name}")

        action = model(np.expand_dims(input_obs, 0)).numpy().argmax()

        prev_obs = obs
        obs, reward, done, info = env.step(action)
        counter += 1

    output_video.close()
    print("Successfully saved {} frames into {}!".format(counter, filename))
    return filename


def save_video_of_memory(memory, filename, size=(512,512)):
    import skvideo.io
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    output_video = skvideo.io.FFmpegWriter(filename)

    for observation in memory.observations:
        output_video.writeFrame(cv2.resize(255*observation, size))
        
    output_video.close()
    return filename

def vista_step(curvature=None, speed=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    car.step_dynamics(action=np.array([curvature, speed]), dt=1/15.)
    car.step_sensors()
    
def upload_data():
  import wget
  url = 'https://www.dropbox.com/s/62pao4mipyzk3xu/vista_traces.zip'
  filename = wget.download(url)
  return filename

def vista_reset():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    world.reset()
    display.reset()
    
class VideoStream():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    import shutil, os, subprocess, cv2
    def __init__(self):
        self.tmp = "./tmp"
        if os.path.exists(self.tmp) and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp)
        os.mkdir(self.tmp)
    def write(self, image, index):
        cv2.imwrite(os.path.join(self.tmp, f"{index:04}.png"), image)
    def save(self, fname):
        cmd = f"/usr/bin/ffmpeg -f image2 -i {self.tmp}/%04d.png -crf 0 -y {fname}"
        subprocess.call(cmd, shell=True)
        
def get_human_trace(time):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    vista_reset()
    stream = VideoStream()
    for i in tqdm(range(time)):
        vista_step()
        vis_img = display.render()
        stream.write(vis_img[:, :, ::-1], index=i)
        if car.done: 
            break  

def check_out_of_lane(car):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width 
    half_road_width = road_width / 2
    return distance_from_center > half_road_width

def check_exceed_max_rot(car):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    maximal_rotation = np.pi / 10.
    current_rotation = np.abs(car.relative_state.yaw)
    return current_rotation > maximal_rotation

def check_crash(car): 
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    return check_out_of_lane(car) or check_exceed_max_rot(car) or car.done

def preprocess(full_obs):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]
    obs = obs / 255.
    return obs

def grab_and_preprocess_obs(car):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    return obs

def create_driving_model():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    model = tf.keras.models.Sequential([
        Conv2D(filters=32, kernel_size=5, strides=2),
        Conv2D(filters=64, kernel_size=3, strides=2),
        Conv2D(filters=64, kernel_size=3, strides=2), 
        Flatten(),
        Dense(units=128, activation=act),
        Dense(units=2, activation=None) 
    ])
    return model

def run_driving_model(image):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    single_image_input = tf.rank(image) == 3  # missing 4th batch dimension
    if single_image_input:
        image = tf.expand_dims(image, axis=0)
    distribution = driving_model(image) # TODO
    mu, logsigma = tf.split(distribution, 2, axis=1)
    mu = max_curvature * tf.tanh(mu) # conversion
    sigma = max_std * tf.sigmoid(logsigma) + 0.005 

    pred_dist = tfp.distributions.Normal(mu, sigma) # TODO
    return pred_dist


def compute_driving_loss(dist, actions, rewards):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    neg_logprob = -1 * dist.log_prob(actions)
    loss = tf.reduce_mean( neg_logprob * rewards )
    return loss

def normalize(x):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

def discount_rewards(rewards, gamma=0.95):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)

def train_step(model, loss_function, optimizer, observations, actions, discounted_rewards, custom_fwd_fn=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
        if custom_fwd_fn is not None:
            prediction = custom_fwd_fn(observations)
        else: 
            prediction = model(observations)
        loss = loss_function(prediction, actions, discounted_rewards) # TODO
    grads = tape.gradient(loss, model.trainable_variables) # TODO
    grads, _ = tf.clip_by_global_norm(grads, 2)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    

def run_experiments():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    learning_rate = 5e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    vista_reset()
    driving_model = create_driving_model()
    smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')
    memory = Memory()
    max_batch_size = 300
    max_reward = float('-inf')
    if hasattr(tqdm, '_instances'): tqdm._instances.clear()
    for i_episode in range(200):
        plotter.plot(smoothed_reward.get())
        vista_reset()
        memory.clear()
        observation = grab_and_preprocess_obs(car)

        while True:
            curvature_dist = run_driving_model(observation)
            curvature_action = curvature_dist.sample()[0,0]
            vista_step(curvature_action)
            observation = grab_and_preprocess_obs(car)
            reward = 1.0 if not check_crash(car) else 0.0
            memory.add_to_memory(observation, curvature_action, reward)
            if reward == 0.0:
                total_reward = sum(memory.rewards)
                smoothed_reward.append(total_reward)
                batch_size = min(len(memory), max_batch_size)
                i = np.random.choice(len(memory), batch_size, replace=False)
                train_step(driving_model, compute_driving_loss, optimizer, 
                                   observations=np.array(memory.observations)[i],
                                   actions=np.array(memory.actions)[i],
                                   discounted_rewards = discount_rewards(memory.rewards)[i], 
                                   custom_fwd_fn=run_driving_model)            
                memory.clear()
                break

def run_test():
    i_step = 5
    num_episodes = 1
    num_reset = 1
    stream = VideoStream()
    for i_episode in range(num_episodes):
        vista_reset()
        observation = grab_and_preprocess_obs(car)
        episode_step = 0
        while not check_crash(car) and episode_step < 100:
            # using our observation, choose an action and take it in the environment
            curvature_dist = run_driving_model(observation)
            curvature = curvature_dist.mean()[0,0]
            vista_step(curvature)
            observation = grab_and_preprocess_obs(car)
            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1], index=i_step)
            i_step += 1
            episode_step += 1
        for _ in range(num_reset):
            stream.write(np.zeros_like(vis_img), index=i_step)
            i_step += 1

    mean_reward = (i_step - (num_reset*num_episodes)) / num_episodes
    return mean_reward

def get_sample_test():
    i_step = 1
    num_episodes = 1
    num_reset = 1
    stream = VideoStream()
    for i_episode in range(num_episodes):
        vista_reset()
        observation = grab_and_preprocess_obs(car)
        episode_step = 0
        while not check_crash(car) and episode_step < 100:
            # using our observation, choose an action and take it in the environment
            curvature_dist = run_driving_model(observation)
            curvature = curvature_dist.mean()[0,0]
            vista_step(curvature)
            observation = grab_and_preprocess_obs(car)
            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1], index=i_step)
            i_step += 1
            episode_step += 1
        for _ in range(num_reset):
            stream.write(np.zeros_like(vis_img), index=i_step)
            i_step += 1

    mean_reward = (i_step - (num_reset*num_episodes)) / num_episodes
    stream.save("trained_policy.mp4")
    return trained_policy.mp4
