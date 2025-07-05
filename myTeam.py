# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import copy
from game import Actions

#################
# Team creation #
#################
#Our two teams are OffensiveAgent and DefensiveAgent
def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent', numTraining = 0):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
#Parent Agent for both our teams 
class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    #Assigning Start state of agent
    self.start = gameState.getAgentPosition(self.index)
    #All Dimensions
    self.Layout_height = gameState.data.layout.height
    self.Layout_width = gameState.data.layout.width
    
    #Mid_Value of Dimensions
    self.Layout_width_midpoint = self.Layout_width / 2
    self.Layout_height_midpoint=self.Layout_height / 2

    #Initial Number of Food
    all_Food=len(self.getFood(gameState).asList())
    self.initialFood = all_Food
    
    #Initial Number of Capsules
    self.all_Capsules=len(self.getCapsules(gameState))

    self.last_Known_Location = None
   #Logic shouls be implemnted
    self.Unbounded_food = []
    self.bounded_food = []
  
  # Reference taken from Baseline.py
  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    
    return random.choice(actions)
# Reference taken from Baseline.py
  def getSuccessor(self, gameState, action):
    next_successor = gameState.generateSuccessor(self.index, action)
    position_of_successor = next_successor.getAgentState(self.index).getPosition()
    if position_of_successor != nearestPoint(position_of_successor):
      return next_successor.generateSuccessor(self.index, action)
    else:
      return next_successor
# Reference taken from Baseline.py
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
# Reference taken from Baseline.py
  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features
# Reference taken from Baseline.py
  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def distance(self,gameState,target):
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    opponents=self.getOpponents(gameState)
    opponents_states_list=[]
    opponents_agent=[]
    opponents_ghost=[]
    
    for opp in opponents:
        opponents_states_list.append(gameState.getAgentState(opp))
    for i in opponents_states_list:
        if i.isPacman :
            opponents_agent.append(i)
        if not i.isPacman and i.getPosition() !=None:
            opponents_ghost.append(i)
    if target=="MY_HOME":
        dis_list=[]
        validPositions=self.non_wall_Coordinates(gameState)
        for validPosition in validPositions:
            dis =  self.getMazeDistance(validPosition,myPosition)
            dis_list.append(dis)
        min_dis=min(dis_list)
        return min_dis   
    
    if target=="CLOSE_GHOST":
        if len(opponents_ghost)>0:
            dis_ghos_list=[]
            for gho in opponents_ghost:
                dis_ghos =  self.getMazeDistance(gho.getPosition(),myPosition)
                dis_ghos_list.append(dis_ghos)
            return min(dis_ghos_list)
        else:
            return None
    
    if target=="ALL_GHOST":
        if len(opponents_ghost)>0:
            dummy_dis=999999
            for all_gho in opponents_ghost:
                dis_all_ghos =  self.getMazeDistance(all_gho.getPosition(),myPosition)
                if dis_all_ghos<dummy_dis:
                    dummy_dis=dis_all_ghos
                    ghostState = all_gho
            li_ghost=[dummy_dis,ghostState]
            return li_ghost
    if target=="FOOD":
      if self.initialFood>0:
            foo_dis=999999
            for foo in opponents_ghost:
                foo_distance =  self.getMazeDistance(foo.getPosition(),myPosition)
                if foo_distance<foo_dis:
                    foo_dis=foo_distance
            return foo_dis
  def non_wall_Coordinates(self,gameState):
    ''''
    return a list of positions of boundary
    '''
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    boundaries = []
    if self.red:
      i = self.Layout_width_midpoint - 1
    else:
      i = self.Layout_width_midpoint + 1
    
    for j in range(self.Layout_height):
      boundaries.append((i,j))
    validPositions = []
    for i in boundaries:
      x=int(i[0])
      y=int(i[1])
      if not gameState.hasWall(x,y):
        validPositions.append(i)
    return validPositions
#Null Heuristic
  def nullHeuristic(self,state, gameState):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

  def lastEatenFoodLocation(self,gameState):
       if len(self.observationHistory)>1:
        preObservedLoc=self.getPreviousObservation()
        preFoodDefended=self.getFoodYouAreDefending(preObservedLoc)
        preFoodDefendedList=preFoodDefended.asList()
        currentFoodDefended = self.getFoodYouAreDefending(gameState)
        currentFoodDefendedList = currentFoodDefended.asList()
        if len(preFoodDefendedList) != len(currentFoodDefendedList):
            self.last_Known_Location = [food for food in preFoodDefendedList  if food not in currentFoodDefendedList]


  def bfs(self, problem, gameState, heuristic=nullHeuristic):
    """Search the shallowest nodes in the search tree first."""
    "* YOUR CODE HERE *"
    #Intitialising stating state
    pacman_startpoint=problem.getStartState()
    #To check is starting state is goal or not
    if problem.isGoalState(pacman_startpoint):
        return[]
    #creating queue to store node 
    frontier_node=util.Queue()
    #creating queue to store action
    pacman_action=util.Queue()
    #pushing starting state into queue
    frontier_node.push((pacman_startpoint))
    #pushing empty list in action queue
    pacman_action.push([])
    #list to store explored node
    exploredNode=[]
    
    while True:
        #To check is node queue is empty or not
        if frontier_node.isEmpty():
            return []
        else:
            #popping node queue
            New_node=frontier_node.pop()
            #popping action queue
            action=pacman_action.pop()
            #To check is popped node is in in explored list
            if New_node not in exploredNode:
                # if not in explored list append it
                exploredNode.append(New_node)
                #To check is starting state is goal or not
                if problem.isGoalState(New_node):
                    #if yes then return action
                    return action
                #To take details from neighbour node
                for content in problem.getSuccessors(New_node):
                    NeighbourNode=content[0]
                    FurtherAction=content[1]
                    cost=content[2]
                    #To get updated action 
                    finalAction=action+[FurtherAction]
                    #To push node
                    frontier_node.push((NeighbourNode))
                    #to push final action
                    pacman_action.push(finalAction)
    util.raiseNotDefined()

# A* Algorithm
  def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    #Intitialising stating state
    
    pacman_startpoint=problem.getStartState()
    #To check is starting state is goal or not
    if problem.isGoalState(pacman_startpoint):
        return[]
    #creating pri queue to store node
    frontier_node=util.PriorityQueue()
    #creating pri queue to store action
    pacman_action=util.PriorityQueue()
    frontier_node.push((pacman_startpoint,0),0)
    pacman_action.push([],0)
    #list to store explored node
    exploredNode=[]
    while True:
        if frontier_node.isEmpty():
            return []
        else:
            #popping node and cost queue
            New_node,cost=frontier_node.pop()
            #popping action queue
            action=pacman_action.pop()
            #To check is popped node is in in explored list
            if New_node not in exploredNode  :
                exploredNode.append(New_node)
                
                if problem.isGoalState(New_node):
                    
                    return action
                #To take details from neighbour node
                for content in problem.getSuccessors(New_node):
                    NeighbourNode=content[0]
                    FurtherAction=content[1]
                    Nextcost=content[2]
                    finalAction=action+[FurtherAction]
                    newcostofnode=cost+Nextcost
                    #To finf Heurestic value from Heurestic function
                    Heurestic_value=newcostofnode+heuristic(NeighbourNode, gameState)
                    #To push final action,Heurestic value
                    pacman_action.push(finalAction,Heurestic_value)
                    #To push Neighbour Node ,new cost and Heaustric value as priority
                    frontier_node.push((NeighbourNode,newcostofnode),Heurestic_value)
    util.raiseNotDefined()

  def generalHeuristic(self, state, currentGameState):
    generalHeuristic = 0
    close_ghost=self.distance(currentGameState,"ALL_GHOST") 

    enemiesOpponents=[]

    if close_ghost != None :
      for opponent in self.getOpponents(currentGameState):
        enemiesOpponents.append(currentGameState.getAgentState(opponent))
      ghostAgent=[]
      for opp in enemiesOpponents :
        if opp.getPosition() != None and opp.scaredTimer < 2 and  not opp.isPacman:
          ghostAgent.append(opp)
      lenOfGhostAgent=len(ghostAgent)
      if lenOfGhostAgent > 0 and ghostAgent != None :
        ghostAgentPositions=[]
        for ghost in ghostAgent:
          ghost_pos=ghost.getPosition()
          ghostAgentPositions.append(ghost_pos)
        ghostAgentDists=[]
        for ghostAgentPosition in ghostAgentPositions:
          self_maze_Dist=self.getMazeDistance(state,ghostAgentPosition)
          ghostAgentDists.append(self_maze_Dist)
        ghostAgentDist = min(ghostAgentDists)
        if ghostAgentDist < 2:
          generalHeuristic = pow((5-ghostAgentDist),5)

    return generalHeuristic




#Child class structure
class OffensiveAgent(DummyAgent):
  
  def time_of_scared(self,gameState):
    opponents = self.getOpponents(gameState)
    for opp in opponents:
      tim=gameState.getAgentState(opp).scaredTimer
      if tim > 1:
        return tim

    return 0
  def chooseAction(self, gameState):
    
    
    present_state=gameState.getAgentState(self.index)
    
    
    
    condition_1=gameState.getAgentState(self.index).numCarrying
    condition_2=len(self.getFood(gameState).asList())
    condition_3=gameState.data.timeleft
    condition_4=self.distance(gameState,"CLOSE_GHOST")
    condition_5=self.distance(gameState,"ALL_GHOST")
    condition_6=self.distance(gameState,"MY_HOME")
    condition_7=self.time_of_scared(gameState)
    condition_8=self.getCapsules(gameState)
    condition_9=self.distance(gameState,"FOOD")

    if condition_3<120 and condition_1>0:
      problem = All_inone_problem(gameState, self, self.index,problem_type=3)
      solution=DummyAgent.aStarSearch(self, problem, gameState, heuristic=self.generalHeuristic)


      if len(solution) == 0:
        return 'Stop'
      else:
        return solution[0]
    

    if len(condition_8)>0:
      if condition_7<8:
        if condition_5 != None:
          if condition_5[1].scaredTimer < 4:
            problem = All_inone_problem(gameState, self, self.index,problem_type=4)

            solution=self.aStarSearch( problem, gameState, heuristic=self.generalHeuristic)

            if len(solution) == 0:
              return 'Stop'
            else:
              
              return solution[0]

    if condition_1<1 :
        problem = All_inone_problem(gameState, self, self.index,problem_type=1)


        solution=DummyAgent.aStarSearch(self, problem, gameState, heuristic=self.generalHeuristic)

        if len(solution) == 0:
            return 'Stop'
        else:
            return solution[0]

    if condition_5 != None:
      if condition_5[0]< 7:
        if condition_5[1].scaredTimer < 4:
          problem = All_inone_problem(gameState, self, self.index,problem_type=2)

          solution=DummyAgent.aStarSearch(self, problem, gameState, heuristic=self.generalHeuristic)

          if len(solution) == 0:
            return 'Stop'
          else:
            
            return solution[0]


    if condition_2 < 3 :
      problem = All_inone_problem(gameState, self, self.index,problem_type=3)
      solution=DummyAgent.aStarSearch(self, problem, gameState, heuristic=self.generalHeuristic)

      
      if len(solution) == 0:
        return 'Stop'
      else:
        return solution[0]
    if condition_3 <  condition_6 :
      problem = All_inone_problem(gameState, self, self.index,problem_type=3)

      solution=DummyAgent.aStarSearch(self, problem, gameState, heuristic=self.generalHeuristic)

      
      if len(solution) == 0:
        return 'Stop'
      else:
        return solution[0]
    if condition_1 > 12:
      problem = All_inone_problem(gameState, self, self.index,problem_type=3)

      solution=DummyAgent.aStarSearch(self, problem, gameState, heuristic=self.generalHeuristic)

      
      if len(solution) == 0:
        return 'Stop'
      else:
        return solution[0]
      



    problem = All_inone_problem(gameState, self, self.index,problem_type=1)

    solution=DummyAgent.aStarSearch(self, problem, gameState, heuristic=self.generalHeuristic)

    if len(solution) == 0:
        return 'Stop'
    else:
        return solution[0]
    
class DefensiveAgent(DummyAgent):
     #This method will choose the actions to be taken and is overridden method for dummy agent when agant is  DefensiveAgent
    def chooseAction(self, currentGameState):
      self.lastEatenFoodLocation(currentGameState)  
      legalActions = currentGameState.getLegalActions(self.index)
      actionValues=[]
      for action in legalActions:
        actionValues.append(self.evaluate(currentGameState, action))
      enemiesAgent=[]
      for enemy in self.getOpponents(currentGameState):
       enemiesAgent.append(currentGameState.getAgentState(enemy))
      invaderAgent_known=[]
      for e_agent in enemiesAgent:
       if e_agent.isPacman and e_agent.getPosition() !=None:
        invaderAgent_known.append(e_agent)
      invaderAgent=[]
      for p_agent in enemiesAgent :
         if p_agent.isPacman:
          invaderAgent.append(p_agent)   
      if len(invaderAgent) == 0:
       self.last_Known_Location = None

      if currentGameState.getAgentPosition(self.index) == self.last_Known_Location:
        self.last_Known_Location = None
      
      if len(invaderAgent_known) > 0:
        self.last_Known_Location = None

      len_invaders=len(invaderAgent)
      len_known_invaders=len(invaderAgent_known)
      scaredTImer=currentGameState.getAgentState(self.index).scaredTimer
      lastKnownFoodLoc=self.last_Known_Location

      if scaredTImer== 0:
        if lastKnownFoodLoc!=None:
            if len_known_invaders== 0:
              searchLastEatenFoodProblem =  All_inone_problem(currentGameState, self, self.index,problem_type=6)
              actionSolution= DummyAgent.bfs(self, searchLastEatenFoodProblem, currentGameState, heuristic=self.generalHeuristic)
              if len(actionSolution) == 0:
                return 'Stop'
              else:
                return actionSolution[0]
      
      if len_known_invaders > 0 :
        if scaredTImer == 0: 
          searchInvaderProblem =  All_inone_problem(currentGameState, self, self.index,problem_type=5)
          actionSolution= DummyAgent.bfs(self, searchInvaderProblem, currentGameState, heuristic=self.generalHeuristic)
          if len(actionSolution) == 0:
                  return 'Stop'
          else:
                  return actionSolution[0]
      

      maxValue = max(actionValues)
      return random.choice([a for a, v in zip(legalActions, actionValues) if v == maxValue])
      
    def getFeatures(self, currentGameState, action):
      features = util.Counter()
      successor = self.getSuccessor(currentGameState, action)

      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      # Computes whether we're on defense (1) or offense (0)
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0

      # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      features['numInvaders'] = len(invaders)
      if len(invaders) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)

      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[currentGameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1

      return features

    def getWeights(self, currentGameState, action):
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}



class PositionSearchProblem():
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState,agent, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True,problem_type=0):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class All_inone_problem(PositionSearchProblem):

 
   
   def __init__(self, gameState, agent, agentIndex = 0,problem_type=0,type=0):
    
    
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        self.lastKnownLocation=agent.last_Known_Location
        
        self.problem_type=problem_type
        self.ourBoundary = agent.non_wall_Coordinates(gameState)
        self.enemies = [gameState.getAgentState(agentIndex) for agentIndex in agent.getOpponents(gameState)]
        self.invaders = [a for a in self.enemies if a.isPacman and a.getPosition != None]
        if len(self.invaders) > 0:
           self.invadersPosition =  [invader.getPosition() for invader in self.invaders]
        else:
           self.invadersPosition = None
       
       
    
   def getStartState(self):
    return self.startState
   def isGoalState(self, state):
    # the goal state is the position of food or capsule
    # return state in self.food.asList() or state in self.capsule
    if self.problem_type==1:
        return state in self.food.asList()
    elif self.problem_type==2:
        return  state in self.ourBoundary or  state in self.capsule 
    elif self.problem_type==3:
        return state in self.ourBoundary
    elif self.problem_type==4:
        return state in self.capsule 
    elif self.problem_type==5:
        return state in self.invadersPosition
    elif self.problem_type==6:
        return state in self.lastKnownLocation
    
