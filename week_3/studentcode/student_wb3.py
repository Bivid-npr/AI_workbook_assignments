
from approvedimports import *

class DepthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "depth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """void in superclass
        In sub-classes should implement different algorithms
        depending on what item it picks from self.open_list
        and what it then does to the openlist

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        
        # my_index <-- GetLastIndex(open_list)
        my_index = len(self.open_list) - 1

        # the_candidate <-- open_list(my_index)
        next_soln = self.open_list[my_index]

        # RemoveFromOpenList(my_index)
        del self.open_list[my_index]
        
        # <==== insert your pseudo-code and code above here
        return next_soln

class BreadthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "breadth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements the breadth-first search algorithm

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here

        # my_index <-- GetFirstIndex(open_list)
        my_index = 0

        # the_candidate <-- open_list(my_index)
        next_soln = self.open_list[my_index]

        # RemoveFromOpenList(my_index)
        del self.open_list[my_index]

        # <==== insert your pseudo-code and code above here
        return next_soln

class BestFirstSearch(SingleMemberSearch):
    """Implementation of Best-First search."""

    def __str__(self):
        return "best-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements Best First by finding, popping and returning member from openlist
        with best quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here

        # IF isEmpty(open_list) THEN
        #    RETURN None
        # ELSE
        #   bestChild <- GetMemberWithHighestQuality(openList)
        #   RETURN bestChild 

        if not self.open_list:
            return None
        else:
            best_index = 0
            for i in range(len(self.open_list)):
                if self.open_list[i].quality < self.open_list[best_index].quality:
                    best_index = i
            next_soln = self.open_list.pop(best_index) 

        # <==== insert your pseudo-code and code above here
        return next_soln

class AStarSearch(SingleMemberSearch):
    """Implementation of A-Star  search."""

    def __str__(self):
        return "A Star"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements A-Star by finding, popping and returning member from openlist
        with lowest combined length+quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here

    # IF isEmpty(open_list) THEN
        #    RETURN None
        # ELSE
        #   bestChild <- GetMemberWithHighestCombinedScore(openList)
        #   RETURN bestChild 

        if not self.open_list:
            return None
        else:
            best_index = 0
            for i in range(len(self.open_list)):
                if self.open_list[i].quality + len(self.open_list[i].variable_values) < self.open_list[best_index].quality + len(self.open_list[best_index].variable_values):
                    best_index = i
            next_soln = self.open_list.pop(best_index) 

        # <==== insert your pseudo-code and code above here
        return next_soln
    
wall_colour= 0.0
hole_colour = 1.0

def create_maze_breaks_depthfirst():
    # ====> insert your code below here
    #remember to comment out any mention of show_maze() before you submit your work

    # Create a default 21x21 maze from maze.txt
    maze = Maze(mazefile="maze.txt")
    
    # Set start and goal positions
    maze.start = (1, 1)
    maze.goal = (19, 19)
    
    # Ensure a clear path from (1,1) to (1,19) to (19,19)
    maze.contents[1][1] = hole_colour
    maze.contents[1][2] = hole_colour
    maze.contents[1][3] = hole_colour
    maze.contents[1][4] = hole_colour
    maze.contents[1][5] = hole_colour
    maze.contents[1][6] = hole_colour
    maze.contents[1][7] = hole_colour
    maze.contents[1][8] = hole_colour
    maze.contents[1][9] = hole_colour
    maze.contents[1][10] = hole_colour
    maze.contents[1][11] = hole_colour
    maze.contents[1][12] = hole_colour
    maze.contents[1][13] = hole_colour
    maze.contents[1][14] = hole_colour
    maze.contents[1][15] = hole_colour
    maze.contents[1][16] = hole_colour
    maze.contents[1][17] = hole_colour
    maze.contents[1][18] = hole_colour
    maze.contents[1][19] = hole_colour
    maze.contents[2][19] = hole_colour
    maze.contents[3][19] = hole_colour
    maze.contents[4][19] = hole_colour
    maze.contents[5][19] = hole_colour
    maze.contents[6][19] = hole_colour
    maze.contents[7][19] = hole_colour
    maze.contents[8][19] = hole_colour
    maze.contents[9][19] = hole_colour
    maze.contents[10][19] = hole_colour
    maze.contents[11][19] = hole_colour
    maze.contents[12][19] = hole_colour
    maze.contents[13][19] = hole_colour
    maze.contents[14][19] = hole_colour
    maze.contents[15][19] = hole_colour
    maze.contents[16][19] = hole_colour
    maze.contents[17][19] = hole_colour
    maze.contents[18][19] = hole_colour
    maze.contents[19][19] = hole_colour
    
    # Long dead-end from (1,15) to (20,15) to trap DFS
    maze.contents[1][15] = hole_colour  # Already set above
    maze.contents[2][15] = hole_colour
    maze.contents[3][15] = hole_colour
    maze.contents[4][15] = hole_colour
    maze.contents[5][15] = hole_colour
    maze.contents[6][15] = hole_colour
    maze.contents[7][15] = hole_colour
    maze.contents[8][15] = hole_colour
    maze.contents[9][15] = hole_colour
    maze.contents[10][15] = hole_colour
    maze.contents[11][15] = hole_colour
    maze.contents[12][15] = hole_colour
    maze.contents[13][15] = hole_colour
    maze.contents[14][15] = hole_colour
    maze.contents[15][15] = hole_colour
    maze.contents[16][15] = hole_colour
    maze.contents[17][15] = hole_colour
    maze.contents[18][15] = hole_colour
    maze.contents[19][15] = hole_colour
    maze.contents[20][15] = hole_colour
    
    # Save the maze
    maze.save_to_txt("maze-breaks-depth.txt")
    # <==== insert your code above here
