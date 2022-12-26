# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            values_i = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    self.values[state] = self.mdp.getReward(state, 'exit', '')
                    continue
                actions = self.mdp.getPossibleActions(state)
                values_i[state] = max([self.computeQValueFromValues(state, action) for action in actions])
            self.values = values_i


    def computeValue(self, state, iteration, max_interation):
        if iteration == max_interation:
            return 0

        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return 0
        action_values = []
        for action in actions:
            sum = 0
            states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
            for state_and_probs in states_and_probs:
                prob = state_and_probs[1]
                state_stripe = state_and_probs[0]
                reward = self.mdp.getReward(state, action, state_stripe)
                value = self.computeValue(state_stripe, iteration+1, max_interation)
                result = prob * (reward + self.discount * value)
                sum = sum + result
            action_values.append(sum)
        max_value = max(action_values)
        return max_value


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        for [new_state, p] in states_and_probs:
            reward = self.mdp.getReward(state, action, new_state)
            value += reward + self.discount*(self.values[new_state]*p)
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        state_action = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            state_action[action] = self.computeQValueFromValues(state, action)
        policy = state_action.argMax()
        return policy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0
        for i in range(self.iterations):
            # In the first iteration, only update the value of the first state in the states list.
            # In the second iteration, only update the value of the second.
            # Keep going until you have updated the value of each state once,
            # then start back at the first state for the subsequent iteration.
            # If the state picked for updating is terminal, nothing happens in that iteration.
            states = self.mdp.getStates()
            index = i % len(states)
            state = states[index]
            if self.mdp.isTerminal(state):
                self.values[state] = self.mdp.getReward(state, 'exit', '')
                continue
            actions = self.mdp.getPossibleActions(state)
            self.values[state] = max([self.getQValue(state, action) for action in actions])

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states.
        states = self.mdp.getStates()
        fringe = util.PriorityQueue()
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            # Find the absolute value of the difference between the current value of s in self.values
            # and the highest Q-value across all possible actions from s
            val = self.getValue(state)
            highest_q = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            diff = abs(val - highest_q)
            print("diff", diff)
            # Push s into the priority queue with priority -diff (note that this is negative).
            fringe.push(state, -diff)

        for iteration in range(self.iterations-1):
            if fringe.isEmpty():
                break
            state = fringe.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                predecessors = self.mdp.getTransitionStatesAndProbs(state, action)
                for [predecessor, prob] in predecessors:
                    if self.mdp.isTerminal(predecessor):
                        continue
                    val = self.values[predecessor]
                    highest_q = max([self.getQValue(predecessor, p_action) for p_action in self.mdp.getPossibleActions(predecessor)])
                    diff = abs(val - highest_q)
                    if diff > self.theta:
                        fringe.update(predecessor, -diff)
