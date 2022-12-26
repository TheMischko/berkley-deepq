# multiAgents.py
# --------------
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
import time

from util import manhattanDistance
from game import Directions
import random, util, queue

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodCount = successorGameState.getNumFood()

        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = successorGameState.getGhostPositions()
        ghostDistances = [util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]
        distToFood = distToClosestFood(successorGameState, newPos[0], newPos[1])
        distToEnemy = distToClosesEnemy(successorGameState)

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore() + (10/distToFood) + distToEnemy*2


def distToClosestFood(gameState, ax, ay):
    map = gameState.getFood().data
    yBounds = (1, len(map[0])-1)
    xBounds = (1, len(map)-1)
    for i in range(1, 20):
        # Top and bottom edge check
        for j in range(max(ax - i, xBounds[0]), min(ax + i, xBounds[1])):
            if map[j][min(ay + i, yBounds[1])]:
                return i
            if map[j][max(ay - i, yBounds[0])]:
                return i
        # Right and left edge check
        for j in range(max(ay - i, yBounds[0]), min(ay + i, yBounds[1])):
            if map[min(ax + i, xBounds[1])][j]:
                return i
            if map[max(ax - i, xBounds[0])][j]:
                return i

    return -1


def distToClosesEnemy(gameState):
    newPos = gameState.getPacmanPosition()
    ghostPositions = gameState.getGhostPositions()
    ghostDistances = [util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]
    return min(ghostDistances)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    pos = currentGameState.getPacmanPosition()
    distToFood = distToClosestFood(currentGameState, pos[0], pos[1])
    distToEnemy = distToClosesEnemy(currentGameState)

    "*** YOUR CODE HERE ***"
    return currentGameState.getScore()
    return currentGameState.getScore() + (10 / distToFood) + distToEnemy * 2


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maxDepth = self.depth
        bestAction = MiniMax(0, gameState, 0, gameState.getNumAgents(), self.evaluationFunction, maxDepth)
        return bestAction


def MiniMax(curDepth, gameState, agentIndex, maxAgents, scoreFnc, maxDepth):
    if curDepth == maxDepth:
        return scoreFnc(gameState)
    actions = gameState.getLegalActions(agentIndex)
    if len(actions) == 0:
        return scoreFnc(gameState)
    if agentIndex == 0:
        scores = [MiniMax(
                    curDepth+1,
                    gameState.generateSuccessor(agentIndex, action),
                    maxAgents-1,
                    maxAgents,
                    scoreFnc,
                    maxDepth)
                for action in actions]
        if(curDepth == 0):
            index = scores.index(max(scores))
            return actions[index]
        return max(scores)
    else:
        return min([
            MiniMax(
                curDepth,
                gameState.generateSuccessor(agentIndex, action),
                agentIndex - 1,
                maxAgents,
                scoreFnc,
                maxDepth)
            for action in actions])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxDepth = self.depth
        alpha = float("-inf")
        beta = float("inf")
        bestAction = AlphaBeta(0, gameState, 0, gameState.getNumAgents(), self.evaluationFunction, maxDepth, alpha, beta)
        return bestAction


def AlphaBeta(curDepth, gameState, agentIndex, maxAgents, scoreFnc, maxDepth, alpha, beta):
    if curDepth == maxDepth:
        return scoreFnc(gameState)
    actions = gameState.getLegalActions(agentIndex)
    if len(actions) == 0:
        return float("-inf")
    # Max branch
    newAlpha = alpha
    newBeta = beta
    if agentIndex == 0:
        bestVal = float("-inf")
        action = None
        for action in actions:
            val = AlphaBeta(
                curDepth + 1,
                gameState.generateSuccessor(agentIndex, action),
                maxAgents - 1,
                maxAgents,
                scoreFnc,
                maxDepth, newAlpha, newBeta)
            bestVal = max(bestVal, val)
            newAlpha = max(newAlpha, bestVal)
            if newAlpha > newBeta:
                break
        if(curDepth == 0):
            print(bestVal)
            return action
        return bestVal
    # Min branch
    else:
        bestVal = float("inf")
        for action in actions:
            val = AlphaBeta(
                curDepth,
                gameState.generateSuccessor(agentIndex, action),
                agentIndex - 1,
                maxAgents,
                scoreFnc,
                maxDepth, newAlpha, newBeta)
            bestVal = min(bestVal, val)
            newBeta = min(newBeta, bestVal)
            if newAlpha > newBeta:
                break
        return bestVal



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()
        maxDepth = self.depth
        bestMove = self.MiniMax(0, gameState, 0, float("-inf"), float("inf"), maxDepth, numAgents)
        return bestMove

    def MiniMax(self, agentIndex, state, depth, alpha, beta, maxDepth, numAgents):
        newAlpha = alpha
        newBeta = beta
        newDepth = depth
        if depth == maxDepth and agentIndex == 0:
            return self.evaluationFnc(state)
        if agentIndex == numAgents - 1:
            newDepth += 1

        legalActions = state.getLegalActions(agentIndex)
        if len(legalActions) == 0:
            return self.evaluationFnc(state)

        nextAgent = (agentIndex + 1) % numAgents

        if agentIndex == 0:
            maxValue = float("-inf")
            bestMove = None
            for action in legalActions:
                newState = state.generateSuccessor(agentIndex, action)
                value = self.MiniMax(nextAgent, newState, newDepth, newAlpha, newBeta, maxDepth, numAgents)
                if value >= maxValue:
                    maxValue = value
                    bestMove = action
                newAlpha = max(newAlpha, maxValue)
                if newBeta < newAlpha:
                    break
            if depth == 0:
                return bestMove
            return maxValue
        else:
            minValue = float("inf")
            for action in legalActions:
                newState = state.generateSuccessor(agentIndex, action)
                value = self.MiniMax(nextAgent, newState, newDepth, newAlpha, newBeta, maxDepth, numAgents)
                value = value / len(legalActions)
                minValue = min(minValue, value)
                newBeta = min(newBeta, value)
                if newBeta < newAlpha:
                    break
            return minValue

    def evaluationFnc(self, currentGameState):
        """
              Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
              evaluation function (question 5).
              DESCRIPTION: <write something here so we know what you did>
            """
        "*** YOUR CODE HERE ***"
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = currentGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Get the current score of the successor state
        score = currentGameState.getScore()

        ghostValue = 10.0
        foodValue = 10.0
        scaredGhostValue = 50.0  # bigger value for the scared ghost because we want to prefer it as a move

        # For every ghost
        for x in newGhostStates:
            # Find the distance from pacman
            dis = manhattanDistance(newPos, x.getPosition())
            if dis > 0:
                """
                If the ghost is edible, and the ghost is near, the distance
                is small.In order to get a bigger score we divide the distance to a big number
                to get a higher score
                """
                if x.scaredTimer > 0:
                    score += scaredGhostValue / dis
                else:
                    score -= ghostValue / dis
                """
                If the ghost is not edible, and the ghost is far, the distance
                is big. We want to avoid such situation so we subtract the distance to a big number
                to lower the score and avoid this state.
                """

        # Find the distance of every food and insert it in a list using manhattan
        foodList = newFood.asList()
        foodDistances = []
        """
        If the food is very close to the pacman then the distance is small and 
        we want such a situation to proceed. So we divide the distance to a big number
        to get a higher score 
        """
        for x in foodList:
            foodDistances.append(manhattanDistance(newPos, x))

        # If there is at least one food
        if len(foodDistances) is not 0:
            score += foodValue / min(foodDistances)

        # Return the final Score
        return score


def betterEvaluationFunction(gameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    return gameState.getScore()


# Abbreviation
better = betterEvaluationFunction
