from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()
    
    # ====> insert your code below here
    options = puzzle.value_set
    for i in options:
        for j in options:
            for k in options:
                for l in options:
                    my_attempt.variable_values = [i,j,k,l]
                    try:
                        if puzzle.evaluate(my_attempt.variable_values):
                            return my_attempt.variable_values
                    except Exception as e:
                        print(e)
    # <==== insert your code above here
    
    # should never get here
    return [-1, -1, -1, -1]
