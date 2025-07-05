# AI' -Project - Pacman Capture the Flag

The purpose of this project is to implement a Pacman Autonomous Agent that can play and compete in the AI'2022   _Pacman Capture the Flag tournament_:

 <p align="center"> 
    <img src="img/logo-capture_the_flag.png" alt="logo project 2" width="400">
  
 </p>
 
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
### Basic rules & guidelines





more time. 
   * Each member will design and record their _own_ video presentation, probably putting emphasis on what the member has contributed. Please include name and student number at the start of the presentation.
   * Showing live demos to depict different implementations, techniques, challenges, etc. is often a good idea in the presentations. 
   * The video will be shared with us through an unlisted youtube link in the Wiki of your GitHub repository.
3. A filled [Project Certification & Contribution Form (FINAL)](https://forms.gle/c3VPYzqKhwmJMZh87).
    * Each member of the team should fill a separate certification form. Members who do not certify will not be marked and will be awarded zero marks.
    * You will reflect on the team contribution with respect to the codebase, report, and video.

**IMPORTANT:** As can be seen by their weighting, the report and video are important components of the project. We strongly recommend working on them *during the development of the project and your system*, for example, by collecting data, screenshots, videos, notes, observations, etc. that may be relevant and potentially useful. Do not leave these components to the last minute, as you may not have enough time to put them together at a high-quality.

## 3. Pre-contest feedback contests

We will be running **_feedback_ contests** based on preliminary versions of teams' agents in the weeks before the final project submission. We will start once **five teams** have submitted their preliminary agents by tagging their repos with "`testing`".

Participating in these pre-contests will give you **a lot of insights** on how your solution is performing and how to improve it. Results, including replays for every game, will be available only for those teams that have submitted. 

You can re-submit multiple times, and we will just run the version tagged `testing`. These tournaments carry no marking at all; they are just designed for **continuous feedback** for you to  analyse and improve your solution! You do not need to certify these versions.

We will try to run these pre-competitions frequently, at least once a day once enough teams are submitting versions.

The top-8 will enter into a playoff series to play quarterfinals, semi-finals and finals, time permitting live in the last day of class or in week 13 in a day specified for that (these final phases will not be part of the marking criteria, just bonus marks).

Additional technical details can be found in [CONTEST.md](CONTEST.md). 

## 5. Inter-University Competition

The top teams of the final tournament will be inducted to the [RMIT-UoM Pacman Hall of Fame](https://sites.google.com/view/pacman-capture-hall-fame/) and will qualify to the yearly championship across RMIT and The University of Melbourne, which runs every year with the best teams since 2017 onward (given you grant us permission, of course). This is just "for fun" and will attract no marks, but is something that previous students have stated in their CVs!

## 8. Conclusion

This is the end of the project assessment specification. Remember to also read the [CONTEST.md](CONTEST.md) file containing technical information that will come very useful (including chocolate prizes for the winners!).



**GOOD LUCK & HAPPY PACMAN!**

Sebastian

### Acknowledgements

This is [Pacman Capture the Flag Contest](http://ai.berkeley.edu/contest.html) from the set of [UC Pacman Projects](http://ai.berkeley.edu/project_overview.html). I am very grateful to UC Berkeley CS188 for developing and sharing their system with us and to RMIT for  learning purposes.
