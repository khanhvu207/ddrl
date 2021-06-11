import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed=2021, hidden_size1=256, hidden_size2=128, hidden_size3=64):
		super(ActorNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, hidden_size1)
		self.fc2 = nn.Linear(hidden_size1, hidden_size2)
		# self.fc3 = nn.Linear(hidden_size2, hidden_size3)
		self.fc4 = nn.Linear(hidden_size2, action_size)
		self.cov_var = torch.full(size=(action_size,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var).to(device)

	def forward(self, state, action=None):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		# x = F.relu(self.fc3(x))
		x = torch.tanh(self.fc4(x))
		dist = torch.distributions.MultivariateNormal(x, self.cov_mat)
		if action is None:
			action = dist.sample()
		log_prob = dist.log_prob(action)
		return action, log_prob

class CriticNetwork(nn.Module):
	def __init__(self, state_size, seed=2021, hidden_size1=256, hidden_size2=128, hidden_size3=64):
		super(CriticNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, hidden_size1)
		self.fc2 = nn.Linear(hidden_size1, hidden_size2)
		# self.fc3 = nn.Linear(hidden_size2, hidden_size3)
		self.fc4 = nn.Linear(hidden_size2, 1)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		# x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x