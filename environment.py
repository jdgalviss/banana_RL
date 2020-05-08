from unityagents import UnityEnvironment
import numpy as np

FILENAME = "Banana.x86_64"
class BananaEnvironment:
    def __init__(self, env_folder, num_previous_frames = 3, train_mode = True):
        self.env_base = UnityEnvironment(file_name = env_folder + '/' +FILENAME)
        self.brain_name = self.env_base.brain_names[0]
        self.brain = self.env_base.brains[self.brain_name]
        self.train_mode = train_mode
        self.nframes = num_previous_frames
        self.last_states = []
        self.env_folder = env_folder
        self.reset()
        if (env_folder == 'VisualBanana_Linux'):
            self.state_size = self.state.shape
        else:
            self.state_size = len(self.state)
            
    def update_state(self):
        if self.env_folder == 'VisualBanana_Linux':
            # state size is 1,84,84,3
            # Rearrange from NHWC to NCHW
            current_state = np.transpose(self.env_info.visual_observations[0], (0,3,1,2))[:,:,:,:]
            frame_size = current_state.shape
            if(len(self.last_states) < self.nframes+1):
                self.last_states.append(current_state)
            else:
                self.last_states.pop(0)
                self.last_states.append(current_state)
            self.state = np.zeros((1, frame_size[1], self.nframes+1, frame_size[2], frame_size[3]))
            for i in range(len(self.last_states)):
                self.state[0, :, i, :, :] = self.last_states[len(self.last_states)-1-i]
        else:
            self.state = self.env_info.vector_observations[0]

    def reset(self):
        self.env_info = self.env_base.reset(train_mode=self.train_mode)[self.brain_name]
        self.update_state()
        return self.state

    def render(self):
        pass

    def step(self, action):
        self.env_info = self.env_base.step(action)[self.brain_name]  # send the action to the environment
        self.update_state()
        reward = self.env_info.rewards[0]  # get the reward
        done = self.env_info.local_done[0]  # see if episode has finished
        return self.state, reward, done, None #info is none

    def close(self):
        self.env_base.close()