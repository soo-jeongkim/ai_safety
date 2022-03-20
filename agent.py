class QNet(torch.nn.Module):
    """
    A network that estimates Q values
    
    Q(state, action) = optimal value of the future

    The heart of a DQN agent is a network that takes a state, and returns
    the estimated future value for each outcome expected for a given
    action.
    """

    def __init__(
      self, 
      hidden_size = 100,
    ):
      super().__init__()

      self.num_actions = 5
      self.input_size = 6 * 7

      self.linear1 = torch.nn.Linear(self.input_size, hidden_size)
      self.relu = torch.nn.ReLU()
      self.linear2 = torch.nn.Linear(hidden_size, self.num_actions)

    def forward(self, x):
      x = torch.flatten(torch.tensor(x))
      return self.linear2(self.relu(self.linear1(torch.reshape(x, (x.shape[0] // self.input_size, self.input_size)))))

class Agent:
    def __init__(self, history_size = 1000, batch_size = 128, gamma=0.9):

        # A history of tuples of format:
        # (state, taken action, reward received, next state, 
        #  whether the episode ended)
        self.history = deque()
        self.history_size = history_size
        self.batch_size = batch_size

        # A neural network that predicts 'Q values' for different states
        self.net = QNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=3e-2)

        # the probabilitiy with which the agent acts randomly. This enables the
        # agent to explore. When it decays, the agent flips to exploitation.
        self.eps = 0.5
        self.eps_min = 0.05
        self.eps_decay = 0.999
        self.updated = False

        self.loss_history = []

        self.gamma = gamma

    def act(self, state):
        """
        Given state (of the gridworld), return an action (indexed between 1 and 5) to take.
        """
        state = torch.flatten(torch.tensor(state))
        
        action = np.random.randint(5)
        return action ### TODO EXERCISE 2: IMPLEMENT EPSILON GREEDY INSTEAD OF THE ABOVE RANDOM CHOICE

    def update_net(self):
        """Update the network
        
        It is trained to predict the value of the whole future for an
        action taken given a state:
            Q(state, action).

        The *actual* information used in the update is the 'reward',
        the 'next state' reached, and an 'action' taken from the
        previous 'state'.

        This is added to the discounted reward for the maximum predicted
        Q value of the next state:
            max_a [ Q(next_state, a) ]
        """
        self.opt.zero_grad()

        sample_indices = np.random.choice(len(self.history), self.batch_size)
        sample = [self.history[i] for i in sample_indices]
        states, actions, rewards, next_states, dones = tuple(zip(*sample))
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        # What is the value of the whole future, from the next state?
        future_q = torch.max(self.net(next_states), axis=1)[0]
        future_q = torch.where(torch.tensor(dones), torch.zeros_like(future_q), future_q)

        # The target, formed of the real reward + prediction for the next state
        q_targets = rewards + self.gamma * future_q

        # What the network predicts currently
        q_predictions_all_acts = self.net(states)

        # sampled at the actions that were taken to obtain the above 
        # rewards, next states
        indices =\
            torch.tensor(list(range(self.batch_size))) * q_predictions_all_acts.shape[1]\
            + actions

        q_predictions_all_acts = torch.flatten(q_predictions_all_acts)
        q_predictions = torch.gather(
          q_predictions_all_acts,
          0, 
          indices
        )

        # get the MSE loss
        loss = torch.mean((q_targets - q_predictions) ** 2)
        self.loss_history.append(loss.detach().clone().item())

        # run gradient descent!
        loss.backward()
        self.opt.step()

def agent_play(env = get_new_env(), agent = Agent(), print_things = True):
    time_step = env.reset()
    
    observation = time_step.observation["board"]
    steps_done = 0

    while True:
        if print_things: print_game(convert_board_num2str(observation))

        if time_step.step_type == environment.StepType.LAST:
            if path is None: print("GAME OVER")
            break

        else:
            current_state = time_step.observation["board"]
            user_input = valid_actions[agent.act(current_state)][0]
            if print_things: print(f"The agent chooses input {user_input}")
        # validate input
        if user_input not in valid_action_strings:
            print("Try an action in", valid_action_strings.keys())
            continue
        elif user_input == "q":
            break

        # env takes an integer action - convert input
        action = valid_action_strings[user_input]

        # take the action in the environment
        time_step = env.step(action)
        observation = time_step.observation["board"]

sample_env = get_new_env(level=0)
sample_env._max_iterations = 200 # you can increase the maximum number of iterations allowed in a given environment
agent_play(env = sample_env, agent=Agent(), print_things=True)
if sample_env._episode_return != -100: print(sample_env._episode_return)
print(f"Agent's final reward: {sample_env._episode_return}")

