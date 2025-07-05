# AI' -Project - Pacman Capture the Flag

The purpose of this project is to implement a Pacman Autonomous Agent that can play and compete in the AI'2022   _Pacman Capture the Flag tournament_:
 
 
Note that the Pacman tournament has different rules as it is a game of two teams, where your Pacmans become ghosts in certain areas of the grid. Please read carefully the rules of the Pacman tournament. Understanding it well and designing a controller for it is part of the expectations for this project. Additional technical information on the contest project can be found in file [CONTEST.md](CONTEST.md). 



## 1.  Task Description

**Task** is to develop an autonomous Pacman agent team to play the [Pacman Capture the Flag Contest](http://ai.berkeley.edu/contest.html) by suitably modifying file `myTeam.py` (and possibly some other auxiliary files). The code submitted  is  internally commented at high standards and be error-free and _never crash_. 

In my approachm  I used at **least 2 AI-related techniques** (**3 techniques at least for groups of 4**) and  combined them in any form. Some candidate techniques that you may consider are:

1. Heuristic Search Algorithms (using general or Pacman specific heuristic functions).
2. Classical Planning (PDDL and calling a classical planner).
3. Value Iteration (Model-Based MDP).
4. Monte Carlo Tree Search or UCT (Model-Free MDP).
5. Reinforcement Learning – classical, approximate or deep Q-learning (Model-Free MDP).
6. Goal Recognition techniques (to infer intentions of opponents).
7. Game Theoretic Methods.
8. Bayesian inference.

We  can always use hand coded decision trees to express behaviour specific to Pacman, but they won't count as a required technique. We  are allowed to express domain knowledge, but remember that we are interested in "autonomy", and hence using techniques that generalise well. The 7 techniques mentioned above can cope with different rules much easier than any decision tree (if-else rules). If we decide to compute a policy, we can save it into a file and load it at the beginning of the game, as we have 15 seconds before every game to perform any pre-computation.

Together with  actual code solution, we developed a Wiki report, documenting and describing  solution (both what ended up in the final system and what didn't), as well as a 5-min recorded video demonstrating your work in the project. 

Additional technical details can be found in [CONTEST.md](CONTEST.md). 

## 5. Inter-University Competition

The top teams of the final tournament will be inducted to the [RMIT-UoM Pacman Hall of Fame](https://sites.google.com/view/pacman-capture-hall-fame/) and will qualify to the yearly championship across RMIT and The University of Melbourne, which runs every year with the best teams since 2017 onward (given you grant us permission, of course). This is just "for fun" and will attract no marks, but is something that previous students have stated in their CVs!

## 8. Conclusion

This is the end of the project assessment specification. Remember to also read the [CONTEST.md](CONTEST.md) file containing technical information that will come very useful (including chocolate prizes for the winners!).


### Acknowledgements

This is [Pacman Capture the Flag Contest](http://ai.berkeley.edu/contest.html) from the set of [UC Pacman Projects](http://ai.berkeley.edu/project_overview.html). I am very grateful to UC Berkeley CS188 for developing and sharing their system with us and to RMIT for  learning purposes.
