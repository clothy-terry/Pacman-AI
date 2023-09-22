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

        # print("successor is")
        # print(successorGameState)
        # print("new pos:")
        # print(newPos)
        # print("new food")
        # print(newFood)
        # print("new ghost state")
        # print(newGhostStates)
        # print("new scared times")
        # print(newScaredTimes)

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        ghostList = successorGameState.getGhostPositions()

        if (foodList):
            foodDist = manhattanDistance(newPos, foodList[0])
            for f in range(len(foodList)):
                newFoodDist = manhattanDistance(newPos, foodList[f])
                if(foodDist>newFoodDist):
                    foodDist = newFoodDist
        else:
            foodDist = 0

        if(newScaredTimes[0]==0):
            ghostDist = manhattanDistance(newPos, ghostList[0])
        else:
            ghostDist = float('inf')
        
        return successorGameState.getScore()+(ghostDist)/10+2/(foodDist+1)


    

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
        v = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            curValue = self.value(successor, 1, self.depth)
            if (curValue>v):
                v = curValue
                bestAction = action
        return bestAction
        


    def value(self, gameState:GameState, agentIndex, depth):
    #Terminal state
        if depth == 0 :
            return self.evaluationFunction(gameState)
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex != 0:
            return self.min(gameState, agentIndex, depth)
        if agentIndex == 0:
            return self.max(gameState, agentIndex, depth)
        else :
            print("error on terminal state")
            
    def max(self, gameState:GameState, agentIndex, depth):
        v = float('-inf')
        
        #iterate through successors
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            if (agentIndex < gameState.getNumAgents() - 1):
                curValue = self.value(successor, agentIndex+1, depth)
            else:
                curValue = self.value(successor, 0, depth-1)
            if (curValue>v):
                v = curValue
        return v

    def min(self, gameState:GameState, agentIndex, depth):
        v = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex < gameState.getNumAgents() - 1):
                curValue = self.value(successor, agentIndex+1, depth)
            else:
                curValue = self.value(successor, 0, depth-1)
            if (curValue<v):
                v = curValue
        return  v
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')
        v = float('-inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            curValue = self.value(successor, 1, self.depth, alpha, beta)
            if (curValue>v):
                v = curValue
                BestAction = action
            alpha = max(alpha, v)
        return BestAction
        util.raiseNotDefined()
        
    def value(self, gameState:GameState, agentIndex, depth, alpha, beta):
    #Terminal state
        if depth == 0 :
            return self.evaluationFunction(gameState)
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex != 0:
            return self.min(gameState, agentIndex, depth, alpha, beta)
        if agentIndex == 0:
            return self.max(gameState, agentIndex, depth, alpha, beta)
        else :
            print("error on terminal state")
            
    def max(self, gameState:GameState, agentIndex, depth, alpha, beta):
        v = float('-inf')
        #iterate through successors
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            if (agentIndex < gameState.getNumAgents() - 1):
                curValue = self.value(successor, agentIndex+1, depth, alpha, beta)
            else:
                curValue = self.value(successor, 0, depth-1, alpha, beta)
            if (curValue>v):
                v = curValue
                if (v>beta):
                    return v
            alpha = max(alpha, v)
        return v
    
    def min(self, gameState:GameState, agentIndex, depth, alpha, beta):
        v = float('inf')
        #iterate through successors
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex < gameState.getNumAgents() - 1):
                curValue = self.value(successor, agentIndex+1, depth, alpha, beta)
            else:
                curValue = self.value(successor, 0, depth-1, alpha, beta)
            if (curValue<v):
                v = curValue
                if (v<alpha):
                    return v
            beta = min(beta, v)
        return v

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
        Maxvalue = float('-inf')
        for action in gameState.getLegalActions(0):
            curSuccessor = gameState.generateSuccessor(0, action)
            curValue = self.value(curSuccessor, 1, self.depth)
            if (curValue > Maxvalue) :
                Maxvalue = curValue
                BestAction = action
        return BestAction
        
    # return a tuple (Score , Action)
    def value(self, gameState:GameState, agentIndex, depth):
        #Terminal state
        if depth == 0 :
            return self.evaluationFunction(gameState)
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex != 0:
            return self.expect_value(gameState, agentIndex, depth)
        elif agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)
        else :
            print("error on terminal state")

    def max_value(self, gameState:GameState,agentIndex, depth):

        Maxvalue = float('-inf')
        for each in gameState.getLegalActions(0):
            curSuccessor = gameState.generateSuccessor(0,each)
            #不考虑没有鬼的情况
            curValue = self.value(curSuccessor, agentIndex + 1, depth)
            if (curValue > Maxvalue) :
                Maxvalue = curValue
        #return the max score at this depth
        return Maxvalue


    def expect_value(self, gameState:GameState, agentIndex, depth):
        value = 0
        for each in gameState.getLegalActions(agentIndex):
            curSuccessor = gameState.generateSuccessor(agentIndex,each)
            if (agentIndex < gameState.getNumAgents() - 1):
                curValue = self.value(curSuccessor, agentIndex + 1, depth)
            else:
                curValue = self.value(curSuccessor, 0, depth - 1)
            value += curValue/len(gameState.getLegalActions(agentIndex))
        return value




def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    powerPellets = currentGameState.getCapsules()
    
    foodList = currFood.asList()
    ghostList = currentGameState.getGhostPositions()
    aggregateFoodDist = 0

    if (foodList):
        aggregateFoodDist = min([manhattanDistance(currPos, f) for f in foodList])
        
    powerScore = 0
    if powerPellets:
        superDist = min(manhattanDistance(currPos, pellet) for pellet in powerPellets)
        powerScore = 20/(superDist+0.01)
    
    
    ghostDist = manhattanDistance(currPos, ghostList[0])
    if(currScaredTimes[0]==0):
        return currentGameState.getScore()*2+(ghostDist)/2+\
                2/(aggregateFoodDist+0.01)+powerScore
    else:
        return currentGameState.getScore()*2+2/(ghostDist+0.01)+\
                2/(aggregateFoodDist+0.01)+powerScore
        

# Abbreviation
better = betterEvaluationFunction
