import numpy as np

###if you want to play 2048 in terminal the code is commented out at the bottom

class Game:
    
    def __init__(self):
        self.board = np.empty((4,4))
        self.board.fill(0)

        startrow2 = np.random.randint(4)
        startrow4 = np.random.randint(4)
    
        startcolumn2 = np.random.randint(4)
        startcolumn4 = np.random.randint(4)

        while startcolumn2 == startcolumn4 and startrow2 == startrow4:
            if startcolumn2 <= 2:
                startcolumn4= startcolumn2+1
            else:
                startcolumn4= startcolumn2-1

        self.board[startcolumn2][startrow2] = 2
        self.board[startcolumn4][startrow4] = 4

        self.score = 0
        self.gameover = False

    def check_pairs(self):
        for p in range(4):
            for y in range(4):
                if p + 1 <= 3:
                    if self.board[p][y] == self.board[p+1][y]:
                        return True
                if p - 1 >= 0:
                    if self.board[p][y] == self.board[p-1][y]:
                        return True
                if y + 1 <= 3:
                    if self.board[p][y] == self.board[p][y+1]:
                        return True
                if y - 1 >= 0:
                    if self.board[p][y] == self.board[p][y-1]:
                        return True
        return False

    def random_new(self):
        if 2048 in self.board:
            self.gameover = True
            print('winner')
            self.score = self.score +10000
        i = 0
        twoandfour = [2,2,2,2,2,2,2,2,2,4]
        whichnum = twoandfour[np.random.randint(10)]
        m = 0    
        while i == 0:
            newrow = np.random.randint(4)
            newcolumn = np.random.randint(4)
            if self.board[newcolumn][newrow] == 0:
                self.board[newcolumn][newrow] = whichnum
                i = 1
            m = m+1
            if m > 50:
                are_pairs = self.check_pairs()
                if are_pairs == True:
                    break
                for i in range(4):
                    for j in range (4):
                        if self.board[i][j] == 0:
                           self.board[i][j] = whichnum
                           break
                self.gameover = True
                break

    def up(self):
        oldboard = self.board.copy()
        changed = []
        for i in range(1,4):
            for j in range(4):
                if self.board[i][j] != 0:
                    p = 0
                    n = 1
                    while i-n >= 0 and self.board[i-n][j] == 0:
                        self.board[i-n][j] = self.board[i-p][j]
                        self.board[i-p][j] = 0 
                        n = n + 1
                        p = p + 1
                    if i+n >= 0 and self.board[i-n][j] != 0 and self.board[i-n][j] == self.board[i-p][j] and [i-n,j] not in changed:
                        self.board[i-n][j] = self.board[i-n][j] *2
                        self.score =  self.score + self.board[i-n][j] *2
                        self.board[i-p][j] = 0
                        changed.append([i-n,j]) 
        if not np.array_equal(oldboard,self.board):                   
            self.random_new()

    def down(self):
        oldboard = self.board.copy()
        changed = []
        for i in [2,1,0]:
            for j in range(4):
                if self.board[i][j] != 0:
                    p = 0
                    n = 1
                    while i+n <= 3 and self.board[i+n][j] == 0:
                        self.board[i+n][j] = self.board[i+p][j]
                        self.board[i+p][j] = 0 
                        n = n + 1
                        p = p + 1
                    if i+n <= 3 and self.board[i+n][j] != 0 and self.board[i+n][j] == self.board[i+p][j] and [i+n,j] not in changed:
                        self.board[i+n][j] = self.board[i+n][j] *2
                        self.score = self.score + self.board[i+n][j] *2
                        self.board[i+p][j] = 0 
                        changed.append([i+n,j])             
        if not np.array_equal(oldboard,self.board):                   
            self.random_new()

    def right(self):
        oldboard = self.board.copy()
        changed = []
        for j in [3,2,1,0]:
            for i in range(4):
                if self.board[i][j] != 0:
                    p = 0
                    n = 1
                    while j+n <= 3 and self.board[i][j+n] == 0:
                        self.board[i][j+n] = self.board[i][j+p]
                        self.board[i][j+p] = 0
                        n = n + 1
                        p = p + 1
                    if j+n <= 3 and self.board[i][j+n] != 0 and self.board[i][j+n] == self.board[i][j+p] and [i,j+n] not in changed:
                        self.board[i][j+n] = self.board[i][j+n] *2
                        self.score =  self.score + self.board[i][j+n] *2
                        self.board[i][j+p] = 0
                        changed.append([i,j+n])              
        if not np.array_equal(oldboard,self.board):                   
            self.random_new()

    def left(self):
        oldboard = self.board.copy()
        changed = []
        for j in range(1,4):
            for i in range(4):
                if self.board[i][j] != 0:
                    p = 0
                    n = 1
                    while j-n >= 0 and self.board[i][j-n] == 0:
                        self.board[i][j-n] = self.board[i][j-p]
                        self.board[i][j-p] = 0
                        n = n + 1
                        p = p + 1
                    if j+n >= 0 and self.board[i][j-n] != 0  and self.board[i][j-n] == self.board[i][j-p] and [i,j-n] not in changed:
                        self.board[i][j-n] = self.board[i][j-n] *2
                        self.score =  self.score + self.board[i][j-n] *2
                        self.board[i][j-p] = 0
                        changed.append([i,j-n])
        if not np.array_equal(oldboard,self.board):                   
            self.random_new()


'''
name = input("enter name to start")
a = Game()

print(a.board)

while a.gameover == False:
    move = input('use wasd keypad to move')
    if not move:
        continue
    print(' ')
    print(' ')
    print(' ')
    print(f'{name} score: {a.score}')
    if 'w' in move[0]:
        a.up()
    if 'a' in move[0]:
        a.left()
    if 's' in move[0]:
        a.down()
    if 'd' in move[0]:
        a.right()
    print(a.board)
    print(' ')
    print(' ')
    print(' ')
    
    if 'stop' in move:
        break
    move = 0


'''
