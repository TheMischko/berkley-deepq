import time

import numpy as np
import torch
from IPython import display
from matplotlib import pyplot as plt

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math

from neural.neuralNet import NeuralNet, get_optimizer, get_criterion
from neural.replay_memory import ReplayMemory, Transition


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()
        self.Q_values = util.Counter()
        self.states = set()
        self.time_visited = util.Counter()

        self.states = set()

    def getQValue(self, state, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0
        return max([self.getQValue(state, action) for action in actions])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        stateBestQValue = self.computeValueFromQValues(state)
        bestActions = [action for action in actions if self.getQValue(state, action) == stateBestQValue]
        policy = random.choice(bestActions)
        return policy

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.getLegalActions(state)

        "*** YOUR CODE HERE ***"
        is_random = util.flipCoin(self.epsilon)
        if is_random:
            return random.choice(legal_actions)
        action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # TP Q Learning
        best_action = self.computeValueFromQValues(nextState)
        estimate = self.getQValue(state, action) + self.alpha * (
                reward + self.discount * best_action - self.getQValue(state, action)
        )
        self.Q_values[(state, action)] = estimate

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        featureVector = self.featExtractor.getFeatures(state, action)
        sum = 0
        for feature in featureVector.keys():
            sum += featureVector[feature] * self.weights[feature]
        return sum

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        q_current_state = self.getQValue(state, action)
        feature_vector = self.featExtractor.getFeatures(state, action)

        q_next_state = self.computeValueFromQValues(nextState)
        difference = (reward + self.discount * q_next_state) - q_current_state
        for feature in feature_vector.keys():
            self.weights[feature] = self.weights[feature] + self.alpha * difference * feature_vector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


class DeepQAgent(ReinforcementAgent):
    BATCH_SIZE = 32
    LR = 1e-4
    UPDATE_PREVIOUS_STEPS = 100
    TERMINAL_TOKEN = 'TERMINAL_STATE'

    def __init__(self, mdp, **args):
        ReinforcementAgent.__init__(self, **args)
        #self.states = mdp.getStates()
        #self.possibleActions = set()
        #for state in self.states:
        #    for action in mdp.getPossibleActions(state):
        #        if action not in self.possibleActions:
        #            self.possibleActions.add(action)
        #self.possibleActions = list(self.possibleActions)
        self.possibleActions = ['arm-down', 'arm-up', 'hand-down', 'hand-up']

        self.previous_net = NeuralNet(2, len(self.possibleActions))
        self.policy_net = NeuralNet(2, len(self.possibleActions))
        self.optimizer = get_optimizer(self.policy_net, learning_rate=self.LR)
        self.criterion = get_criterion()
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        self.epsilon = 1.
        self.loss_arr = []
        self.lengths_arr = []
        self.current_length = 0
        self.fig, (self.ax_loss, self.ax_length) = plt.subplots(1, 2)
        self.ax_loss.set_title('Loss')
        self.ax_loss.set_xlabel('Iterations')
        self.ax_loss.set_ylabel('Q-function loss')

        self.ax_length.set_title('Steps until reward')
        self.ax_length.set_xlabel('Episodes')
        self.ax_length.set_ylabel('Steps')

        self.fig.show()



    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        input = torch.Tensor(state)
        values = self.policy_net.forward(input)
        return values[self.possibleActions.index(action)].item()

    def computeActionFromQValues(self, state):
        legal_actions = self.getLegalActions(state)
        action_values = [self.getQValue(state, action) for action in legal_actions]
        policy = legal_actions[np.argmax(action_values)]
        print(policy)
        return policy

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        is_random = util.flipCoin(self.epsilon)
        legal_actions = self.getLegalActions(state)
        if is_random:
            return random.choice(legal_actions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) \
            if next_state != self.TERMINAL_TOKEN \
            else self.TERMINAL_TOKEN
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Store observation in memory
        self.memory.push(
            state,
            torch.tensor(self.possibleActions.index(action)).unsqueeze(0).unsqueeze(0),
            next_state,
            torch.tensor(reward).unsqueeze(0))
        self.learn_from_experience()
        self.steps_done += 1
        self.current_length += 1

        if reward > 0:
            self.lengths_arr.append(self.current_length)
            self.current_length = 0

        if self.epsilon > 0.2:
            print("---EPSILON---", self.epsilon)
            self.epsilon -= 0.0002

        if self.steps_done % self.UPDATE_PREVIOUS_STEPS == 0:
            print("---COPYING WEIGHTS---", self.epsilon)
            prev_net_weights = self.previous_net.state_dict()
            policy_net_weights = self.policy_net.state_dict()
            for key in policy_net_weights:
                prev_net_weights[key] = policy_net_weights[key]
        if self.steps_done % 100 == 0:
            self.ax_loss.plot(self.loss_arr)
            self.ax_length.plot(self.lengths_arr)
            #plt.pause(0.05)

    def learn_from_experience(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch
        # This converts batch-array of Transitions to Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        # Compute mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not self.TERMINAL_TOKEN, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([state for state in batch.next_state if state is not self.TERMINAL_TOKEN])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q value
        # The model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net.
        # Selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in case of
        # terminal state.
        next_state_values = reward_batch
        with torch.no_grad():
            next_state_values[non_final_mask] = next_state_values[non_final_mask] + self.discount * self.previous_net(non_final_next_states).max(1)[0]

        # Compute loss
        loss = self.criterion(state_action_values, next_state_values.unsqueeze(1))
        print("loss", loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.loss_arr.append(loss.item())

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        plt.plot(self.loss_arr)
        plt.pause(0.05)


    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
