import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) # V??? tr?? kh???i t???o c???a c??c box
    beginPlayer = PosOfPlayer(gameState) # V??? tr?? kh???i t???o player

    startState = (beginPlayer, beginBox) # Tr???ng th??i ban ?????u game bao g???m v??? tr?? box v?? player 
    frontier = collections.deque([[startState]]) # S??? d???ng deque l??u tr???ng th??i c???a game
    exploredSet = set() # D??ng set l??u c??c tr???ng th??i ???? th???c hi???n
    actions = [[0]] # D??ng l??u h??nh ?????ng
    temp = []
    while frontier:
        node = frontier.pop() # ????a node cu???i deque ra x??? l?? 
        node_action = actions.pop() # ????a node_actions cu???i deque ra x??? l?? 
        if isEndState(node[-1][-1]): # Ki???m tra v??? tr?? Posbox c?? th???a m??n kh??ng
            temp += node_action[1:] # N???u th???a m??n l??u k???t qu??? c??c h??nh ?????ng v??o temp
            break # N???u th???a m??n, k???t th??c v??ng l???p
        if node[-1] not in exploredSet: # Ki???m tra node cu???i [(newPosPlayer, newPosBox)] ???? ???????c x??? l?? ch??a
            exploredSet.add(node[-1]) # N???u ch??a x??? l?? th?? add v??o exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Duy???t qua c??c h??nh ?????ng h???p l??? (????y l?? h??? s??? r??? nh??nh c???a ch??ng ta)
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # C???p nh???t v??? tr?? player v?? box
                if isFailed(newPosBox): # Ki???m tra v??? tr?? PosBox c?? h???p l??? hay kh??ng
                    continue # N???u kh??ng h???p l???, th?? b??? qua
                frontier.append(node + [(newPosPlayer, newPosBox)]) # N???u h???p l???, th?? add n?? v??o tr???ng th??i game
                actions.append(node_action + [action[-1]]) # Add th??m c??? h??nh ?????ng c???a n?? v??o tr???ng th??i game
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) # V??? tr?? kh???i t???o c???a c??c box
    beginPlayer = PosOfPlayer(gameState) # V??? tr?? kh???i t???o player

    startState = (beginPlayer, beginBox) # Tr???ng th??i ban ?????u game bao g???m v??? tr?? box v?? player  # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]]) # store states
    actions = collections.deque([[0]]) # store actions
    exploredSet = set() # D??ng set l??u c??c tr???ng th??i ???? th???c hi???n
    temp = []
    ### Implement breadthFirstSearch here
    # ch??? n??y t????ng t??? thu???t to??n DFS, nh??ng thay v?? s??? d???ng FILO, m??nh s??? d???ng FIFO
    while frontier:
        node = frontier.popleft()  # ????a node ?????u deque ra x??? l?? 
        node_action = actions.popleft() # ????a node_actions ?????u deque ra x??? l?? 
        if isEndState(node[-1][-1]):  # Ki???m tra v??? tr?? Posbox c?? th???a m??n kh??ng
            temp += node_action[1:] # N???u th???a m??n l??u k???t qu??? c??c h??nh ?????ng v??o temp
            break # N???u th???a m??n, k???t th??c v??ng l???p
        if node[-1] not in exploredSet: # Ki???m tra node cu???i [(newPosPlayer, newPosBox)] ???? ???????c x??? l?? ch??a
            exploredSet.add(node[-1])  # N???u ch??a x??? l?? th?? add v??o exploredSet
            for action in legalActions(node[-1][0], node[-1][1]):  # Duy???t qua c??c h??nh ?????ng h???p l???
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # C???p nh???t v??? tr?? player v?? box
                if isFailed(newPosBox): # Ki???m tra v??? tr?? PosBox c?? h???p l??? hay kh??ng
                    continue # N???u kh??ng h???p l???, th?? b??? qua
                frontier.append(node + [(newPosPlayer, newPosBox)]) # N???u h???p l???, th?? add n?? v??o tr???ng th??i game
                actions.append(node_action + [action[-1]]) # Add th??m c??? h??nh ?????ng c???a n?? v??o tr???ng th??i game
    return temp

    
def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) # V??? tr?? kh???i t???o c???a c??c box
    beginPlayer = PosOfPlayer(gameState) # V??? tr?? kh???i t???o player

    startState = (beginPlayer, beginBox) # Tr???ng th??i ban ?????u game bao g???m v??? tr?? box v?? player
    frontier = PriorityQueue() # S??? d???ng PriorityQueue l??u tr???ng th??i ban ?????u c???a game
    frontier.push([startState], 0) # L??u tr???ng th??i ban ?????u c???a game v???i th??? t??? ??u ti??n cao nh???t
    exploredSet = set() # S??? d???ng set l??u tr???ng th??i ???? x??? l??
    actions = PriorityQueue() # S??? d???ng PriorityQueue l??u h??nh ?????ng c???a game
    actions.push([0], 0) # L??u h??nh ?????ng ?????u ti??n l?? [0] v???i ????? ??u ti??n  cao nh???t
    temp = []
    ### Implement uniform cost search here
    # ch??? n??y t????ng t??? DFS v?? BFS, thay v?? s??? d???ng deque 
    # ta s??? d???ng class PriorityQueue v???i th??? t??? ??u ti??n c??ng cao th?? c??ng g???n v??? tr?? ban ?????u
    priority = 1
    while frontier:
        priority += 1
        node = frontier.pop()  # ????a node cu???i PriorityQueue ra x??? l?? 
        node_action = actions.pop() # ????a node_action cu???i PriorityQueue ra x??? l?? 
        if isEndState(node[-1][-1]): # Ki???m tra v??? tr?? Posbox c?? th???a m??n kh??ng 
            temp += node_action[1:] # N???u th???a m??n l??u k???t qu??? c??c h??nh ?????ng v??o temp
            break  # N???u th???a m??n, k???t th??c v??ng l???p
        if node[-1] not in exploredSet: # Ki???m tra node cu???i [(newPosPlayer, newPosBox)] ???? ???????c x??? l?? ch??a
            exploredSet.add(node[-1]) # N???u ch??a x??? l?? th?? add v??o exploredSet
            
            for action in legalActions(node[-1][0], node[-1][1]): # Duy???t qua c??c h??nh ?????ng h???p l???
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # C???p nh???t v??? tr?? player v?? box
                if isFailed(newPosBox):  # Ki???m tra v??? tr?? PosBox c?? h???p l??? hay kh??ng
                    continue # N???u kh??ng hopje l??? th?? b??? qua
                frontier.push(node + [(newPosPlayer, newPosBox)], priority) # Th??m v??o PriorityQueue tr???ng th??i hi???n t???i, v???i th??? t??? ??u ti??n t??ng d???n theo ????? d??i h??nh ?????ng
                actions.push(node_action + [action[-1]], priority) # Th??m v??o PriorityQueue h??nh ?????ng hi???n t???i, v???i th??? t??? ??u ti??n t??ng d???n theo ????? d??i h??nh ?????ng
    return temp

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.5f second.' %(method, time_end-time_start))
    print(result)
    return result
