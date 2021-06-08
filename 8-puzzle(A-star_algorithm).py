import queue
class State:
    def __init__(self,block,goal,moves=0):
        self.block=block
        self.moves=moves
        self.goal=goal

    def Moving(self,i1,i2,moves): #블럭 이동 함수
        NewBlock=self.block[:]
        NewBlock[i1],NewBlock[i2] = NewBlock[i2],NewBlock[i1]
        return State(NewBlock,self.goal,moves)

    def expand(self,moves):
        result=[]
        i=self.block.index(0) #빈칸의 위치 찾기
        if not i in [0,1,2]: #Up
            result.append(self.Moving(i,i-3,moves))
        if not i in [0,3,6]: #Left
            result.append(self.Moving(i,i-1,moves))
        if not i in [2,5,8]: #Right
            result.append(self.Moving(i,i+1,moves))
        if not i in [6,7,8]: #Down
            result.append(self.Moving(i,i+3,moves))
        
        return result
         
    def f(self):
        return self.h()+self.g()
    def g(self):
        return self.moves        
    def h(self):
        Hn=0
        for i in range(9):
            if self.block[i] != self.goal[i] and self.block[i]!=0:
              Hn+=1              
        return Hn


    def __lt__(self,other):
        return self.f()<other.f()

    def __str__(self):
        return "f(n)=" + str(self.f())+"\n"+\
        "h(n)=" + str(self.h())+"\n"+\
        "g(n)=" + str(self.g())+"\n"+\
        str(self.block[:3])+"\n"+\
        str(self.block[3:6])+"\n"+\
        str(self.block[6:])+"\n"+\
        "------------"
        
ori = [2,8,3,
        1,6,4,
        7,0,5]
goal = [1,2,3,
        8,0,4,
        7,6,5]


Puzzle_set = State(ori,goal)
open=queue.PriorityQueue()
open.put(Puzzle_set)
closed=[]
moves=0

while not open.empty(): #오픈 큐가 비어있기 전 까지
    current=open.get()
    print(current)
    if current.block==goal:
        print("%d회 이동, 탐색완료" %moves)
        break
    moves = current.moves+1
    for state in current.expand(moves):
        if state not in closed:
            open.put(state)
            closed.append(current)
        else:
            print("탐색실패")