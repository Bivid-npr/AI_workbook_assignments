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
                    if puzzle.evaluate(my_attempt.variable_values):
                        return my_attempt.variable_values
    # <==== insert your code above here
    
    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here
    for each in range(namearray.shape[0]):
        family_name = ''.join(namearray[each,-6:])
        family_names.append(family_name)
    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays
    
    # ====> insert your code below here

    # use assertions to check that the array has 2 dimensions each of size 9
    assert attempt.shape == (9,9), "The array must have 9 rows and 9 columns"


    ## Remember all the examples of indexing above
    ## and use the append() method to add something to a list
    for i in range(9):
        slices.append(attempt[i,:])

    for i in range(9):
        slices.append(attempt[:,i])

    slices.append(attempt[0:3, 0:3])  
    slices.append(attempt[0:3, 3:6])  
    slices.append(attempt[0:3, 6:9])  

    slices.append(attempt[3:6, 0:3])  
    slices.append(attempt[3:6, 3:6])  
    slices.append(attempt[3:6, 6:9])  

    slices.append(attempt[6:9, 0:3])  
    slices.append(attempt[6:9, 3:6])  
    slices.append(attempt[6:9, 6:9])  
    

    for slice in slices:  # easiest way to iterate over list
        # print(slice) - useful for debugging?

        # get number of unique values in slice
        uniques = len(np.unique(slice))

        # increment value of tests_passed as appropriate
        if uniques == 9:
            tests_passed += 1
    
    # <==== insert your code above here
    # return count of tests passed
    return tests_passed
