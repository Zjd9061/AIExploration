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

        
        result = self.minimax(gameState, 0, 0)

        # Return the action from result
        return self.minimax(gameState, 0, 0)[1] #runs minimax with game state only. returns the actions

    def minimax(self, gameState, agentIndex, depth):

        if len(gameState.getLegalActions(agentIndex)) == 0 or depth == self.depth: #if there are no legal actions or if the depth is fully explored
            return gameState.getScore(), "" #return the score of the game
        
        if agentIndex < 1: #max function. For pacman
            return self.maxfunc(gameState, agentIndex, depth)

        else: #min function for ghosts
            return self.minfunc(gameState, agentIndex, depth)
    
    def maxfunc(self, gameState, agentIndex, depth):
        max_score = -float("inf") #initialize max score to negative infinity so any move is better to start
        max_move = "" #initialize to empty action just in case there is no better move, will return something

        # This is where you can see that this is a look ahead agent. It is looking at the successor states and evaluating them
        for action in gameState.getLegalActions(agentIndex): #for legal actions
            succ_index = agentIndex + 1 #successor index is not pacman
            succ_depth = depth #successor depth starts at current depth

            if succ_index == gameState.getNumAgents(): #if there is only pacman and one ghost
                succ_index = 0 #set successor index to pacman
                succ_depth = succ_depth +1 #depth increments by 1

            score = self.minimax(gameState.generateSuccessor(agentIndex, action), succ_index, succ_depth)[0] #set the score to the score given by minimax function (recursively called) of the current agents, depth, game state

            if score > max_score: #if the score given above is better than previous maximum score
                max_score = score #update maximum score
                max_move = action #update max move
            
        return max_score, max_move 

    def minfunc(self, gameState, agentIndex, depth): #This is very similar to the max function but it simply checks if the score given by minimax is less than the previous minimum score
        min_score = float("inf")
        min_move = ""

        for action in gameState.getLegalActions(agentIndex):
            succ_index = agentIndex + 1
            succ_depth = depth

            if succ_index == gameState.getNumAgents():
                succ_index = 0
                succ_depth = succ_depth + 1

            score = self.minimax(gameState.generateSuccessor(agentIndex, action), succ_index, succ_depth)[0]

            if score < min_score:
                min_score = score
                min_move = action
        
        return min_score, min_move

    # I struggled really hard with understanding how to write this on my own and was searching the internet for a few days for an explanation of how to implement minimax 
    # that made sense to me.
    # I couldn't find anything that I fully comprehended until I came across this github at https://github.com/khanhngg/CSC665-multi-agent-pacman/blob/master/multiagent/multiAgents.py
    # Full disclosure: I heavily used this to understand how a minimax agent is actually written. If this is an issue, please let me know and I would be happy to meet with you.
    # I hope I have demonstrated that with this example to work off of, I now comprehend how a minimax agent can be implemented



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
