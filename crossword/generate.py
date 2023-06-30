from queue import Empty
import sys

from crossword import *
import itertools
import copy

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains:
            inconsistent = set()
            for word in self.domains[var]:
                if len(word) != var.length:
                    inconsistent.add(word)
            
            for word in inconsistent:
                self.domains[var].remove(word)

        


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x,y]

        if overlap is not None:
            
            words_to_remove=set()
            for x_word in self.domains[x]:
                x_letter = x_word[overlap[0]]
                y_letters = set()
                for y_word in self.domains[y]:
                    letter = y_word[overlap[1]]
                    y_letters.add(letter)

                if x_letter not in y_letters:
                    words_to_remove.add(x_word)
                    revised = True
            
            for word in words_to_remove:
                self.domains[x].remove(word)
        
        return revised





        raise NotImplementedError

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        #assign all overlapping arcs to queue if 'arcs' is None
        if arcs is None:
            queue = list(itertools.product(self.crossword.variables, self.crossword.variables))
            queue = [x for x in queue if x[0] != x[1] and self.crossword.overlaps[x[0], x[1]] is not None]
        #use arcs as initial list of arcs to make consistent
        else:
            queue = arcs
                
        #while queue is not emptyu
        while queue:
            
            arc = queue.pop()
            x = arc[0]
            y = arc[1]

            #use revise to make arc consistent

            if self.revise(x, y):

                #if domain is empty return False
                if not self.domains[x]:
                    return False
                
                for a in (self.crossword.neighbors(x) - {y}):
                    queue.append((a, x))
        
        return True

            

        


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        
        if set(assignment.keys()) == self.crossword.variables and all(assignment.values()):
            return True
        else:
            return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        for variable, value in assignment.items():
            for neighbor in self.crossword.neighbors(variable).intersection(assignment.keys()):
                overlap = self.crossword.overlaps[variable, neighbor]
                if value[overlap[0]] != assignment[neighbor][overlap[1]]:
                    return False
        
        if any(variable.length != len(word) for variable, word in assignment.items()):
            return False

        if len(set(assignment.values())) != len(set(assignment.keys())):
            return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        n_of_neighbor_values_deleted = {word: 0 for word in self.domains[var]}

        neighbors = self.crossword.neighbors(var)

        for value in self.domains[var]:
            for neighbor in (neighbors - assignment.keys()):
                overlap = self.crossword.overlaps[var, neighbor]

                for neighbor_value in self.domains[neighbor]:
                    if value[overlap[0]] != neighbor_value[overlap[1]]:
                        n_of_neighbor_values_deleted[value] += 1
        
        list_sorted = sorted(n_of_neighbor_values_deleted.items(), key=lambda x:x[1])
        return [a[0] for a in list_sorted]


        

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        unassigned = self.crossword.variables - assignment.keys()

        number_of_unassigned_variable_values = {variable: len(self.domains[variable]) for variable in unassigned}
        unassigned_variables_ranked = sorted(number_of_unassigned_variable_values.items(), key=lambda x: x[1])
        
        if len(unassigned_variables_ranked) == 1 or unassigned_variables_ranked[0][1] != unassigned_variables_ranked[1][1]:
            return unassigned_variables_ranked[0][0]
        else:
            number_of_neighbors = {variable: len(self.crossword.neighbors(variable)) for variable in unassigned}
            sorted_by_total_neighbors = sorted(number_of_neighbors.items(), key=lambda x: x[1])
            return sorted_by_total_neighbors[-1][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            test = copy.deepcopy(assignment)
            test[var] = value
            if self.consistent(test):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            assignment.pop(var, None)
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
