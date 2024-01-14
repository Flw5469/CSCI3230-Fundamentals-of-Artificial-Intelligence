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

# If you have any question, ask in Piazza or email me (mxchen21@cse.cuhk.edu.hk)
# DO NOT copy the answer from any website. You can refer to tutorial slides, but try it by yourself first! 


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from sys import path
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

def depthFirstSearch(problem):
    
    """ 
    Question 1: Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    You only need to submit this file. Do not change other files!
    If you finish this function, you almost finish all the questions!
    Read util.py to find suitable data structure!
    All you need is pass all the code in commands.txt
    """

    # SOLUTION 1 iterative function
    
def depthFirstSearch(problem):
    visited = set()  
    stack = util.Stack() 

    startState = problem.getStartState()
    stack.push((startState, []))

    while not stack.isEmpty():
        state, path = stack.pop()

        if problem.isGoalState(state):
            return path

        successors = problem.getSuccessors(state)

        for successor, action, _ in successors:
            if successor not in visited:
                stack.push((successor, path + [action]))
                visited.add(successor);

    return []
    

    # SOLUTION 1 recursive function
    "*** YOUR CODE HERE ***"
    
    
    """
    def dfs(problem,state,counter,direction):
        counter.append(state);
        if (problem.isGoalState(state)):
            return [direction];
        else: 
            for i in problem.getSuccessors(state):
                if (i[0] not in counter):
                    result=dfs(problem,i[0],counter,i[1]);
                    if not (result==[]):
                        if not (direction==None):
                            result.append(direction);
                        return result;
            return [];
    
    
    def recursive_dfs(problem):
        counter=[];
        result=dfs(problem,problem.getStartState(),counter,None);
        result.reverse();
        return result;
    
    return recursive_dfs(problem);
    """
    

def breadthFirstSearch(problem):
    """Question 2: Search the shallowest nodes in the search tree first."""

    "*** YOUR CODE HERE ***"
    def bfs(problem):
        s=util.Queue();
        
        top=(problem.getStartState(),[]);
        s.push(top);
        counter=[];
        counter.append(top[0]);
        while (not (problem.isGoalState(top[0]))) or (s.isEmpty()):
                top=s.pop();
                if (problem.isGoalState(top[0])):
                    result=top[1]
                    return result;
                for i in problem.getSuccessors(top[0]):
                    if (i[0] not in counter):
                        temp=top[1].copy();
                        temp.append(i[1])
                        s.push((i[0],temp));
                        counter.append(i[0]);
        

        

    
    return bfs(problem);


def uniformCostSearch(problem):
    """Question 3: Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"


    def ucs(problem):
        s=util.PriorityQueue();
        top=(problem.getStartState(),[],0);
        s.push((top[0],top[1],top[2]),top[2]);
        counter=[];
        while (not (problem.isGoalState(top[0]))) or (s.isEmpty()):
                top=s.pop();
                if (top[0] not in counter):
                    counter.append(top[0]);
                    if (problem.isGoalState(top[0])):
                        result=top[1]
                        return result;
                    for i in problem.getSuccessors(top[0]):
                            temp=top[1].copy();
                            temp.append(i[1])
                            s.push((i[0],temp,top[2]+i[2]),top[2]+i[2]);
                            
        

        

    
    return ucs(problem);


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Question 4: Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #find new edge, should try lower edge
    def ass(problem):
        s=util.PriorityQueue();
        top=(problem.getStartState(),[],0);
        s.push((top[0],top[1],top[2]),top[2]);
        counter=[];
        while (not (problem.isGoalState(top[0]))) or (s.isEmpty()):
                top=s.pop();
                if (top[0] not in counter):
                    counter.append(top[0]);
                    if (problem.isGoalState(top[0])):
                        result=top[1]
                        return result;
                    for i in problem.getSuccessors(top[0]):
                            temp=top[1].copy();
                            temp.append(i[1]);
                            s.push((i[0],temp,i[2]+top[2]),top[2]+i[2]+heuristic(i[0],problem));
                            
        

        

    answer=ass(problem);
    return answer;


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
