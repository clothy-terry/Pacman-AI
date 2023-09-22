# search.py
# ---------
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

#############################################################################
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    cur = problem.getStartState();
    curNode = cur;
    pathToCur = [];
    res = [];
    visited = [];
    stack = util.Stack();
    stack.push(pathToCur);
    stackOfNode = util.Stack();
    stackOfNode.push(curNode);
    while (not stack.isEmpty()):
        pathToCur = stack.pop();
        curNode = stackOfNode.pop();
        print(pathToCur)
        print(curNode)
        if (problem.isGoalState(curNode)):
            return(pathToCur);
        if (not visited.__contains__(curNode)):
            visited.append(curNode);
        pathSoFar = pathToCur.copy();
        for each in problem.getSuccessors(curNode):
            if (not visited.__contains__(each[0])):
                pathToCur.append(each[1])
                stack.push(pathToCur);
                pathToCur = pathSoFar.copy();
                stackOfNode.push(each[0]);
    return res;

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    cur = problem.getStartState();
    curNode = cur;
    pathToCur = [];
    res = [];
    visited = [];
    queue = util.Queue();
    queue.push([pathToCur, curNode]);
    while (not queue.isEmpty()):
        list = queue.pop();
        pathToCur = list[0];
        curNode = list[1];
        if (problem.isGoalState(curNode)):
            return(pathToCur);
        visited.append(curNode);
        pathSoFar = pathToCur.copy();
        for each in problem.getSuccessors(curNode):
            if (not visited.__contains__(each[0])):
                visited.append(each[0])
                pathToCur.append(each[1])
                queue.push([pathToCur,each[0]]);
                pathToCur = pathSoFar.copy();
    return res;
    util.raiseNotDefined()

def getPriority(l1):
    return l1[2].getCostOfActions(l1[0]);

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    curNode = problem.getStartState();
    pathToCur = [];
    res = [];
    visited = [];
    pq = util.PriorityQueueWithFunction(getPriority);
    pq.push([pathToCur, curNode, problem]);
    while (not pq.isEmpty()):
        list = pq.pop();
        pathToCur = list[0];
        curNode = list[1];
        if (problem.isGoalState(curNode)):
            return(pathToCur);
        visited.append(curNode);
        pathSoFar = pathToCur.copy();
        for each in problem.getSuccessors(curNode):
            if (not visited.__contains__(each[0]) or problem.isGoalState(each[0])):
                visited.append(each[0])
                pathToCur.append(each[1])
                pq.push([pathToCur,each[0],problem]);
                pathToCur = pathSoFar.copy();
    return res;
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

class Node:
    def __init__(self, state, cost):
        self.state = state
        self.cost = cost

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    curNode = Node(start, 0)
    pathToCur = []
    visited = []
    queue = util.PriorityQueue()
    queue.push([pathToCur, curNode], heuristic(start, problem))
    while (not queue.isEmpty()):
        list = queue.pop()
        pathToCur = list[0]
        curNode = list[1]
        if (problem.isGoalState(curNode.state)):
            return(pathToCur)
        if ((curNode.state) not in visited):
            visited.append(curNode.state)
            pathSoFar = pathToCur.copy()
            for each in problem.getSuccessors(curNode.state):
                if ((each[0] not in visited) or problem.isGoalState(each[0])):
                    pathToCur.append(each[1])
                    queue.push([pathToCur, Node(each[0],each[2]+
                        curNode.cost)], each[2]+
                        heuristic(each[0], problem)+curNode.cost)
                    pathToCur = pathSoFar.copy()
    return []
    util.raiseNotDefined()



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
