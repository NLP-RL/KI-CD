

from convlab.agent.memory.base import Memory
from convlab.lib import logger, util
from convlab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


class OnPolicyReplay(Memory):


    def __init__(self, memory_spec, body):
        super().__init__(memory_spec, body)
        # NOTE for OnPolicy replay, frequency = episode; for other classes below frequency = frames
        util.set_attr(self, self.body.agent.agent_spec['algorithm'], ['training_frequency'])
        # Don't want total experiences reset when memory is
        self.is_episodic = True
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.reset()

    @lab_api
    def reset(self):
        '''Resets the memory. Also used to initialize memory vars'''
        for k in self.data_keys:
            setattr(self, k, [])
        self.cur_epi_data = {k: [] for k in self.data_keys}
        self.most_recent = (None,) * len(self.data_keys)
        self.size = 0

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Interface method to update memory'''
        self.add_experience(state, action, reward, next_state, done)

    def add_experience(self, state, action, reward, next_state, done):
        '''Interface helper method for update() to add experience to memory'''
        self.most_recent = (state, action, reward, next_state, done)
        for idx, k in enumerate(self.data_keys):
            self.cur_epi_data[k].append(self.most_recent[idx])
        # If episode ended, add to memory and clear cur_epi_data
        if util.epi_done(done):
            for k in self.data_keys:
                getattr(self, k).append(self.cur_epi_data[k])
            self.cur_epi_data = {k: [] for k in self.data_keys}
            # If agent has collected the desired number of episodes, it is ready to train
            # length is num of epis due to nested structure
            # if len(self.states) == self.body.agent.algorithm.training_frequency:
            if len(self.states) % self.body.agent.algorithm.training_frequency == 0:
                self.body.agent.algorithm.to_train = 1
        # Track memory size and num experiences
        self.size += 1
        self.seen_size += 1

    def get_most_recent_experience(self):
        '''Returns the most recent experience'''
        return self.most_recent

    def sample(self):

        batch = {k: getattr(self, k) for k in self.data_keys}
        self.reset()
        return batch


class OnPolicyBatchReplay(OnPolicyReplay):


    def __init__(self, memory_spec, body):
        super().__init__(memory_spec, body)
        self.is_episodic = False

    def add_experience(self, state, action, reward, next_state, done):
        '''Interface helper method for update() to add experience to memory'''
        self.most_recent = [state, action, reward, next_state, done]
        for idx, k in enumerate(self.data_keys):
            getattr(self, k).append(self.most_recent[idx])
        # Track memory size and num experiences
        self.size += 1
        self.seen_size += 1
        # Decide if agent is to train
        if len(self.states) == self.body.agent.algorithm.training_frequency:
            self.body.agent.algorithm.to_train = 1

    def sample(self):
        
        return super().sample()
