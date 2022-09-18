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


from fileinput import close
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodstuffs = successorGameState.getFood().asList()
        closestFood = 99999 #initialize to large number so any distance to a food will be lower at first
        
        #Go through the food locations and find the closest one
        for food in foodstuffs:
            closestFood = min(closestFood, manhattanDistance(newPos, food))

        #Avoid ghosts
        for blinky in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, blinky) < 3):
                return -99999 #want utility to be very low if near a ghost


        return successorGameState.getScore() + closestFood**-1 #as the food is closer, the utlity is higher

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        #return self.max_agent(gameState, 0)#[0]
        chosen_action = self.minimax(gameState, self.depth)#[1]
        return chosen_action
    
    #def minimax(self, gameState, agentIndex, depth):
    #    if depth is self.depth * gameState.getNumAgents() \
    #            or gameState.isLose() or gameState.isWin():
    #        return self.evaluationFunction(gameState)
    #    if agentIndex is 0:
    #        return self.maxfunc(gameState, agentIndex, depth)#[1]
    #    else:
    #        return self.minfunc(gameState, agentIndex, depth)#[1]

    #def maxval(self, gameState, agentIndex, depth):
    #    bestAction = ("max",-float("inf"))
    #    for action in gameState.getLegalActions(agentIndex):
    #        succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
    #                                  (depth + 1)%gameState.getNumAgents(),depth+1))
    #        bestAction = max(bestAction,succAction,key=lambda x:x[1])
    #    return bestAction

    #def minval(self, gameState, agentIndex, depth):
    #    bestAction = ("min",float("inf"))
    #    for action in gameState.getLegalActions(agentIndex):
    #        succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
    #                                  (depth + 1)%gameState.getNumAgents(),depth+1))
    #        bestAction = min(bestAction,succAction,key=lambda x:x[1])
    #    return bestAction


    #def maxfunc(self, gameState, agentIndex, depth):
    #    maxAction = -float("inf")
    #    for action in gameState.getLegalActions(agentIndex):
    #        successorAction = self.minimax(gameState.generateSuccessor(agentIndex, action), (depth+1)%gameState.getNumAgents(), depth+1)
    #        maxAction = max(maxAction, successorAction)
    #    return maxAction

    #def minfunc(self, gameState, agentIndex, depth):
    #    minAction = -float("inf")
    #    for action in gameState.getLegalActions(agentIndex):
    #        successorAction = self.minimax(gameState.generateSuccessor(agentIndex, action), (depth+1)%gameState.getNumAgents(), depth+1)
    #        minAction = min(minAction, successorAction)
    #    return minAction

    def minimizer(self, game_state, depth, agent):
        actions = game_state.getLegalActions(agent)
        scores = []

        for action in actions:
            successor_game_state = game_state.generateSuccessor(agent, action)

            if agent == game_state.getNumAgents() - 1:  # last ghost
                scores.append(self.minimax(successor_game_state, depth - 1, agent=0, maximizing=True)[0])
            else:
                scores.append(self.minimax(successor_game_state, depth, agent=agent + 1, maximizing=False)[0])

        min_score = min(scores)
        min_indexes = [i for i, score in enumerate(scores) if score == min_score]
        chosen_action = actions[random.choice(min_indexes)]

        return min_score, chosen_action

    def maximizer(self, game_state, depth, agent):
        actions = game_state.getLegalActions(agent)
        scores = []

        for action in actions:
            successor_game_state = game_state.generateSuccessor(agent, action)
            scores.append(self.minimax(successor_game_state, depth, agent=agent + 1, maximizing=False)[0])

        max_score = max(scores)
        max_indexes = [i for i, score in enumerate(scores) if score == max_score]
        chosen_action = actions[random.choice(max_indexes)]

        return max_score, chosen_action
    
    def maxfunc(self, gameState, depth, agentIndex):
        #move = gameState.getLegalActions(agentIndex)
        scores = []

        for action in gameState.getLegalActions(agentIndex):
            successorAction = gameState.generateSuccessor(agentIndex, action)
            scores.append(self.minimax(successorAction, depth, agentIndex + 1, maxormin=False))

            maxAction = max(scores)
        return maxAction

    def minfunc(self, gameState, depth, agentIndex):
        #move = gameState.getLegalActions(agentIndex)
        scores = []

        for action in gameState.getLegalActions(agentIndex):
            successorAction = gameState.generateSuccessor(agentIndex, action)
            scores.append(self.minimax(successorAction, depth, agentIndex + 1, maxormin=False))

            minAction = min(scores)
        return minAction


    def minimax(self, game_state, depth, agentIndex=0, maxormin = True):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)#, Directions.STOP

        if maxormin:
            return self.maxAgent(game_state, depth, agentIndex)
        else:
            return self.minAgent(game_state, depth, agentIndex)

    def minAgent(self, gameState, depth, agentIndex):
        minmoves = []
        for actions in gameState.getLegalActions(agentIndex):
            nextAction = gameState.generateSuccessor(agentIndex, actions)

            if agentIndex == gameState.getNumAgents() - 1:
                minmoves.append(self.minimax(gameState, depth -1, 0, True)[0])
            else:
                minmoves.append(self.minimax(gameState, depth, agentIndex + 1, False)[0])

       # bestMove = min(minmoves)
        return min(minmoves)

    def maxAgent(self, gameState, depth, agentIndex):
        maxmoves = []
        for actions in gameState.getLegalActions(agentIndex):
            nextAction = gameState.generateSuccessor(agentIndex, actions)
            maxmoves.append(self.minimax(gameState, depth, agentIndex + 1, False)[0])

       # bestMove = min(minmoves)
        return max(maxmoves)

        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
