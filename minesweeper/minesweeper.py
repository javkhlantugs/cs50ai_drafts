import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells
        else:
            return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        else:
            return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count = self.count - 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        #mark the cell as a move that has been made
        self.moves_made.add(cell)

        #mark the ce;; as a safe
        self.mark_safe(cell)

        #find all the neighbors of the cell
        neighbors = self.find_the_neighbors(cell)
        mine_count = count

        #check if any of the neighbor cells are known mines or known safes and remove them
        known_neighbors = set()
        for i in neighbors:
            
            if i in self.safes:
                known_neighbors.add(i)
            if i in self.mines:
                known_neighbors.add(i)
                mine_count -= 1
        neighbors -= known_neighbors


        #add unknown neighbor cells sentence to knowledge base
        new_sentence = Sentence(neighbors, mine_count)
        self.knowledge.append(new_sentence)

        #mark any additional cells as safe or as mines if it can be concluded based on the AI's knowledge base
        mines_to_mark = set()
        safes_to_mark = set()
        for sentence in self.knowledge:
            for i in sentence.known_safes():
                safes_to_mark.add(i)
            for i in sentence.known_mines():
                mines_to_mark.add(i)
        for i in mines_to_mark:
            self.mark_mine(i)
        for i in safes_to_mark:
            self.mark_safe(i)

        empty_sentence  = []
        for sentence in self.knowledge:
            if sentence.cells == set():
                empty_sentence.append(sentence)
        for sentence in empty_sentence:
            self.knowledge.remove(sentence)

        #add any new sentences to the AI's knowledge base if they can be inferred from existing knowledge
        inferred_from_existing_knowledge = []
        for sentence in self.knowledge:
            if new_sentence.cells and sentence.cells != new_sentence.cells:
                if new_sentence.cells.issubset(sentence.cells):
                    inferred_from_existing_knowledge.append(Sentence(sentence.cells - new_sentence.cells, sentence.count - new_sentence.count))
                if sentence.cells.issubset(new_sentence.cells):
                    inferred_from_existing_knowledge.append(Sentence(new_sentence.cells - sentence.cells, new_sentence.count - sentence.count))
        self.knowledge += inferred_from_existing_knowledge             

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for safe in self.safes:
            if safe not in self.moves_made:
                return safe
        return None

        raise NotImplementedError

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        entire_board = set(itertools.product(range(self.height), range(self.width)))
        available_random = list(entire_board - self.mines - self.safes - self.moves_made)

        if not available_random:
            return None
        
        return random.choice(available_random)

        raise NotImplementedError


    def find_the_neighbors(self, cell):
        
        i = cell[0]
        j = cell[1]
        neighbors = set()

        #neighbors left right
        if (j-1 >= 0):
            neighbors.add((i,j-1))
        if (j+1 <= self.width - 1):
            neighbors.add((i, j+1))
        
        #neighbors above
        if (i-1 >= 0):
            neighbors.add((i-1, j))
            if (j-1 > -1):
                neighbors.add((i-1, j-1))
            if (j+1 < self.width):
                neighbors.add((i-1, j+1))
        
        #neighbors below
        if (i+1 <= self.height - 1):
            neighbors.add((i+1, j))
            if (j-1 > -1):
                neighbors.add((i+1, j-1))
            if (j+1 < self.width):
                neighbors.add((i+1, j+1))
        
        return neighbors
