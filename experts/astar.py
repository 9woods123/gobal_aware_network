
##created by woods 2023-10-25

from heapq import heappush, heappop
from sys import maxsize
import math


F, H, NUM, G, INDEX, POS, OPEN, PARENT = range(8)
RESOLUTION = 1


class Node:
    def __init__(self, g, h, f, index, pos, is_open, parent_id):
        self.g = g
        self.h = h
        self.f = f
        self.index = index
        self.pos = pos           ## x,y, phsi
        self.is_open = is_open
        self.parent_id = parent_id


class DynamicAstar:

    def __init__(self,resolution:float,radius=1,minStepLength=1,collision_radius=0.5,gird=None) -> None:

        ### forced condition minStepLength>__resolution
        self.__resolution=resolution
        RESOLUTION=resolution
        self.__grid=gird
        self.__reachGoalRadius=radius
        self.minX=0
        self.maxX=25
        self.minY=0
        self.maxY=25
        self.collision_radius=collision_radius

        self.__MotionPrimitives=[(minStepLength,0),
                                            (-minStepLength,0),
                                            (0,minStepLength),
                                            (0,-minStepLength),
                                            (-minStepLength,minStepLength),
                                            (minStepLength,minStepLength),
                                            (-minStepLength,-minStepLength),
                                            (minStepLength,-minStepLength)
                                              ]
        
        self.__goal=None
        self.__start=None
        self.__path=[]
        self.__vel=2

    def setMap(self,grid):
         self.__grid=grid

    def getVel(self):
        return  self.__vel
    
    def setGoal(self,goal:tuple):
        self.__goal=goal
    
    def setStart(self,start:tuple):
        self.__start=start

    def position2Index(self, pos:tuple):
        int_elements = [int(element /  self.__resolution) for element in pos]
        nodeIndex = tuple(int_elements)
        return nodeIndex


    def collision_detection(self,curr_point,next_point):
        return self.__grid.collision_detection(curr_point,next_point, self.collision_radius)
 

    def genNextSearchPoint(self, curr_node):

        x,y = curr_node[POS]
        nextPoints = []

        for motionInput in  self.__MotionPrimitives:
            temp_x=x+motionInput[0]
            temp_y=y+motionInput[1]
            ##  rangeOut check & collision check 
            if not self.collision_detection((x,y),(temp_x,temp_y)):
                        nextPoints.append((temp_x, temp_y))
        

        return nextPoints


    def getGoal(self,pos:tuple):
        d=math.sqrt((pos[0]- self.__goal[0])**2 + (pos[1]- self.__goal[1])**2)
        if d<self.__reachGoalRadius:
            return True
        else:
            return False

    def cost(self,from_pos, to_pos):
        from_x, from_y= from_pos
        to_x,to_y = to_pos

        return math.sqrt( (from_y - to_y)**2
                    +(from_x - to_x)**2)
    
    def heuristic(self,pos):
        x,y = pos
        goal_x, goal_y = self.__goal
        dy, dx = abs(goal_y - y), abs(goal_x - x)
        return dy+dx
    
    def getPathLength(self):
        if len(self.__path)>1:
            return self.__path[-1][G]

    def pathPlan(self, start_g, limit=maxsize):
        self.__path=[]
        # start_pos = self.position2Index(start_pos)
        start_index=self.position2Index(self.__start)
        nums = iter(range(maxsize))
        start_h = self.heuristic(self.__start)
        start = [start_g + start_h, start_h, next(nums), start_g,start_index, self.__start, True, None]
        # F, H, NUM, G, INDEX, POS, OPEN, PARENT

        nodes = {start_index: start}
        heap = [start]
        best = start
        
        while heap:

            current = heappop(heap)
            
            if current[OPEN]==False:
                continue

            current[OPEN] = False

            if self.getGoal(current[POS]):
                best = current
                break

            for neighbor_pos in self.genNextSearchPoint(current):
                neighbor_g = current[G] + self.cost(current[POS], neighbor_pos)
                neighbor_index= self.position2Index(neighbor_pos)
                neighbor = nodes.get(neighbor_index)
                
                if neighbor_index not in nodes:
                    if len(nodes) >= limit:
                        continue
                    neighbor_h = self.heuristic(neighbor_pos)
                    neighbor = [neighbor_g + neighbor_h, neighbor_h, 
                                next(nums), neighbor_g, neighbor_index,
                                neighbor_pos, True, current[INDEX]]
                    nodes[neighbor_index] = neighbor

                    heappush(heap, neighbor)
                    if neighbor_h < best[H]:
                        best = neighbor

                elif neighbor_g < neighbor[G]:

                    if neighbor[OPEN]:

                        neighbor[POS]=neighbor_pos
                        nodes[neighbor_index] = neighbor = neighbor[:]

                        neighbor[F] = neighbor_g + neighbor[H]
                        neighbor[NUM] = next(nums)
                        neighbor[G] = neighbor_g
                        neighbor[PARENT] = current[INDEX]
                        heappush(heap, neighbor)
                    else:
                        neighbor[F] = neighbor_g + neighbor[H]
                        neighbor[G] = neighbor_g
                        neighbor[PARENT] = current[INDEX]
                        neighbor[OPEN] = True
                        heappush(heap, neighbor)


        path = []
        current = best
        while current[PARENT] is not None:
            path.append(current)
            current = nodes[current[PARENT]]
        path.reverse()

        self.__path=path
        wayPoints=[]
        for point in path:
            wayPoints.append(point[POS])

        success=True
        return wayPoints,success,nodes
