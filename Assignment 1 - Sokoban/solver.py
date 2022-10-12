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
    beginBox = PosOfBoxes(gameState) # Vị trí khởi tạo của các box
    beginPlayer = PosOfPlayer(gameState) # Vị trí khởi tạo player

    startState = (beginPlayer, beginBox) # Trạng thái ban đầu game bao gồm vị trí box và player 
    frontier = collections.deque([[startState]]) # Sử dụng deque lưu trạng thái của game
    exploredSet = set() # Dùng set lưu các trạng thái đã thực hiện
    actions = [[0]] # Dùng lưu hành động
    temp = []
    while frontier:
        node = frontier.pop() # Đưa node cuối deque ra xử lí 
        node_action = actions.pop() # Đưa node_actions cuối deque ra xử lí 
        if isEndState(node[-1][-1]): # Kiểm tra vị trí Posbox có thỏa mãn không
            temp += node_action[1:] # Nếu thỏa mãn lưu kết quả các hành động vào temp
            break # Nếu thỏa mãn, kết thúc vòng lặp
        if node[-1] not in exploredSet: # Kiểm tra node cuối [(newPosPlayer, newPosBox)] đã được xử lí chưa
            exploredSet.add(node[-1]) # Nếu chưa xử lí thì add vào exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua các hành động hợp lệ (đây là hệ số rẽ nhánh của chúng ta)
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Cập nhật vị trí player và box
                if isFailed(newPosBox): # Kiểm tra vị trí PosBox có hợp lệ hay không
                    continue # Nếu không hợp lệ, thì bỏ qua
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Nếu hợp lệ, thì add nó vào trạng thái game
                actions.append(node_action + [action[-1]]) # Add thêm cả hành động của nó vào trạng thái game
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) # Vị trí khởi tạo của các box
    beginPlayer = PosOfPlayer(gameState) # Vị trí khởi tạo player

    startState = (beginPlayer, beginBox) # Trạng thái ban đầu game bao gồm vị trí box và player  # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]]) # store states
    actions = collections.deque([[0]]) # store actions
    exploredSet = set() # Dùng set lưu các trạng thái đã thực hiện
    temp = []
    ### Implement breadthFirstSearch here
    # chỗ này tương tự thuật toán DFS, nhưng thay vì sử dụng FILO, mình sử dụng FIFO
    while frontier:
        node = frontier.popleft()  # Đưa node đầu deque ra xử lí 
        node_action = actions.popleft() # Đưa node_actions đầu deque ra xử lí 
        if isEndState(node[-1][-1]):  # Kiểm tra vị trí Posbox có thỏa mãn không
            temp += node_action[1:] # Nếu thỏa mãn lưu kết quả các hành động vào temp
            break # Nếu thỏa mãn, kết thúc vòng lặp
        if node[-1] not in exploredSet: # Kiểm tra node cuối [(newPosPlayer, newPosBox)] đã được xử lí chưa
            exploredSet.add(node[-1])  # Nếu chưa xử lí thì add vào exploredSet
            for action in legalActions(node[-1][0], node[-1][1]):  # Duyệt qua các hành động hợp lệ
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Cập nhật vị trí player và box
                if isFailed(newPosBox): # Kiểm tra vị trí PosBox có hợp lệ hay không
                    continue # Nếu không hợp lệ, thì bỏ qua
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Nếu hợp lệ, thì add nó vào trạng thái game
                actions.append(node_action + [action[-1]]) # Add thêm cả hành động của nó vào trạng thái game
    return temp

    
def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) # Vị trí khởi tạo của các box
    beginPlayer = PosOfPlayer(gameState) # Vị trí khởi tạo player

    startState = (beginPlayer, beginBox) # Trạng thái ban đầu game bao gồm vị trí box và player
    frontier = PriorityQueue() # Sử dụng PriorityQueue lưu trạng thái ban đầu của game
    frontier.push([startState], 0) # Lưu trạng thái ban đầu của game với thứ tự ưu tiên cao nhất
    exploredSet = set() # Sử dụng set lưu trạng thái đã xử lí
    actions = PriorityQueue() # Sử dụng PriorityQueue lưu hành động của game
    actions.push([0], 0) # Lưu hành động đầu tiên là [0] với độ ưu tiên  cao nhất
    temp = []
    ### Implement uniform cost search here
    # chỗ này tương tự DFS và BFS, thay vì sử dụng deque 
    # ta sử dụng class PriorityQueue với thứ tự ưu tiên càng cao thì càng gần vị trí ban đầu
    priority = 1
    while frontier:
        priority += 1
        node = frontier.pop()  # Đưa node cuối PriorityQueue ra xử lí 
        node_action = actions.pop() # Đưa node_action cuối PriorityQueue ra xử lí 
        if isEndState(node[-1][-1]): # Kiểm tra vị trí Posbox có thỏa mãn không 
            temp += node_action[1:] # Nếu thỏa mãn lưu kết quả các hành động vào temp
            break  # Nếu thỏa mãn, kết thúc vòng lặp
        if node[-1] not in exploredSet: # Kiểm tra node cuối [(newPosPlayer, newPosBox)] đã được xử lí chưa
            exploredSet.add(node[-1]) # Nếu chưa xử lí thì add vào exploredSet
            
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua các hành động hợp lệ
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Cập nhật vị trí player và box
                if isFailed(newPosBox):  # Kiểm tra vị trí PosBox có hợp lệ hay không
                    continue # Nếu không hopje lệ thì bỏ qua
                frontier.push(node + [(newPosPlayer, newPosBox)], priority) # Thêm vào PriorityQueue trạng thái hiện tại, với thứ tự ưu tiên tăng dần theo độ dài hành động
                actions.push(node_action + [action[-1]], priority) # Thêm vào PriorityQueue hành động hiện tại, với thứ tự ưu tiên tăng dần theo độ dài hành động
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
