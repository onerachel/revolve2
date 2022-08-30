
# simulation parameters
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 5
SIMULATION_TIME = 10 #50*32

# PPO parameters
PPO_CLIP_EPS = 0.2 # Agrim 1e-5
PPO_LAMBDA = 0.95
PPO_GAMMA = 0.99

# loss weights
ACTOR_LOSS_COEFF = 1
CRITIC_LOSS_COEFF = 0.25
ENTROPY_COEFF = 0.01

# number of steps before each training
NUM_STEPS = 128

# learning rates
LR_ACTOR = 8e-4
LR_CRITIC = 1e-3

#batch size for sgd (M in PPO paper)
BATCH_SIZE = 2048 #Agrim 512
N_EPOCHS = 4

NUM_PARALLEL_AGENT = 64

# number of steps to pass as observations
NUM_OBS_TIMES = 3

# dimension of the different types of  observations
NUM_OBSERVATIONS = 2