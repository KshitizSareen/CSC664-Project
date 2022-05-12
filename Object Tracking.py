import math
import cv2
import numpy as np
import copy

class Point:
    def __init__(self,x,y):
        self.x=x
        self.y=y


def CheckIfOverlapping(l1,r1,l2,r2):
    if l1.x>=r2.x or l2.x>=r1.x:
        return False
    if r1.y<=l2.y or r2.y<=l1.y:
        return False
    
    return True

def CalculateArea(l1,r1,l2,r2):
    xDist=min(r1.x,r2.x)-max(l1.x,l2.x)
    yDist=max(r1.y,r2.y)-min(l1.y,l2.y)
    return abs(xDist*yDist)


def FormGraph(blobsOne,blobsTwo):
    Edges=[]
    blobs=blobsOne+blobsTwo
    Graph=[[0 for i in range(len(blobs))] for j in range(len(blobs))]
    for i in range(len(blobs)):
        for j in range(i+1,len(blobs)):
            blobOne=blobs[i]
            blobTwo=blobs[j]
            if blobOne['frameNumber'] ==1 and blobTwo['frameNumber']==2:
                centroidOne=blobOne['centroid']
                centroidTwo=blobTwo['centroid']
                widthOne=blobOne['boxWidth']
                widthTwo=blobTwo['boxWidth']
                heightOne=blobOne['boxHeight']
                heightTwo=blobTwo['boxHeight']
                areaOne=blobOne['boxArea']
                areaTwo=blobTwo['boxArea']
                l1=Point(centroidOne[0]-widthOne/2,centroidOne[1]-heightOne/2)
                r1=Point(centroidOne[0]+widthOne/2,centroidOne[1]+heightOne/2)
                l2=Point(centroidTwo[0]-widthTwo/2,centroidTwo[1]-heightTwo/2)
                r2=Point(centroidTwo[0]+widthTwo/2,centroidTwo[1]+heightTwo/2)
                if CheckIfOverlapping(l1,r1,l2,r2):
                    l1=Point(centroidOne[0]-widthOne/2,centroidOne[1]+heightOne/2)
                    r1=Point(centroidOne[0]+widthOne/2,centroidOne[1]-heightOne/2)
                    l2=Point(centroidTwo[0]-widthTwo/2,centroidTwo[1]+heightTwo/2)
                    r2=Point(centroidTwo[0]+widthTwo/2,centroidTwo[1]-heightTwo/2)
                    area=CalculateArea(l1,r1,l2,r2)
                    minArea=min(areaOne,areaTwo)
                    if area >= 0.5*minArea:
                        Graph[i][j]=1
                        Graph[j][i]=1
                        Edges.append([i,j])
    
    return Graph,blobs,Edges


class BuildGraph:

    def __init__(self) -> None:
        self.minCost=float("inf")
        self.minGraph=[]

    def DFS(self,nodes,node,visited,Graph):
        visited[node]=True
        nodes.append(node)

        for i in range(len(Graph[node])):
            if Graph[node][i]==1 and visited[i]==False:
                nodes=self.DFS(nodes,i,visited,Graph)

        return nodes


    def ConnectedComponents(self,Graph):

        visited={}
        cc=[]
        
        for node in range(len(Graph)):
            visited[node]=False
        
        for node in range(len(Graph)):
            if visited[node]==False:
                nodes=[]
                cc.append(self.DFS(nodes,node,visited,Graph))
        
        return cc

    def GetDegree(self,node,Graph):
        degree=0
        for i in range(len(Graph[node])):
            if Graph[node][i]==1:
                degree+=1
        return degree

    def CheckValidComponent(self,Component,Graph):
        count=0
        for node in Component:
            degree=self.GetDegree(node,Graph)
            if degree>1:
                if count==1:
                    return False
                count+=1
        return True

    def ComputeGraph(self,Graph,blobs):

        components=self.ConnectedComponents(Graph)

        for component in components:
            if not self.CheckValidComponent(component,Graph):
                return float("inf")
        
        Parents=[]

        for node in range(len(Graph)):
            degree=self.GetDegree(node,Graph)
            if degree>1:
                Parents.append(node)
            elif degree==0:
                Parents.append(node)
            elif degree==1 and blobs[node]['frameNumber']==1:
                Parents.append(node)

        cost=0
        for parent in Parents:
            S=0
            for node in range(len(Graph[parent])):
                if Graph[parent][node]==1:
                    S+=blobs[node]['area']
            area=blobs[parent]['area']
            cost+=((abs(area-S))/(max(area,S)))
        
        return cost


    def RemoveEdges(self,index,Graph,Edges,blobs):
        cost=self.ComputeGraph(Graph,blobs)
        if cost<self.minCost:
            self.minCost=cost
            self.minGraph=copy.deepcopy(Graph)

        for i in range(index,len(Edges)):
            vertexOne=Edges[i][0]
            vertexTwo=Edges[i][1]

            Graph[vertexOne][vertexTwo]=0
            Graph[vertexTwo][vertexOne]=0

            self.RemoveEdges(i+1,Graph,Edges,blobs)

            Graph[vertexOne][vertexTwo]=1
            Graph[vertexTwo][vertexOne]=1


def GetDegree(node,Graph):
    degree=0
    for i in range(len(Graph[node])):
        if Graph[node][i]==1:
            degree+=1
    return degree       

def countEdges(nodes,Graph):
    count=0
    for node in nodes:
        for i in range(len(Graph[node])):
            if Graph[node][i]==1:
                count+=1
    return count//2
    

def GetEdges(Graph):
    Edges=[]
    for i in range(len(Graph)):
        for j in range(len(Graph)):
            if Graph[i][j] == 1:
                if [i,j] not in Edges and [j,i] not in Edges:
                    Edges.append([i,j])
    return Edges

def GetLargestBlob(blobs,Graph,node):
    maxArea=float("-inf")
    largestBlob=None
    for i in range(len(Graph[node])):
        if Graph[node][i] == 1:
            if blobs[i]['area'] > maxArea:
                maxArea=blobs[i]['area']
                largestBlob=blobs[i]
    return largestBlob


def ComputeVelocities(blobs,Graph,Edges,samplingInterval):
    B = 0.5
    for edge in Edges:
        blobOne=edge[0]
        blobTwo=edge[1]
        if GetDegree(blobOne,Graph) > 1:
            blobs[blobTwo]['velocityX'] = blobs[blobOne]['velocityX']
            blobs[blobTwo]['velocityY'] = blobs[blobOne]['velocityY']
        elif GetDegree(blobTwo,Graph) > 1:
            largestBlob=GetLargestBlob(blobs,Graph,blobTwo)
            blobs[blobTwo]['velocityX'] = largestBlob['velocityX']
            blobs[blobTwo]['velocityY'] = largestBlob['velocityY']
        elif GetDegree(blobTwo,Graph) == 1:
            centroidOne = blobs[blobOne]['centroid']
            centroidTwo = blobs[blobTwo]['centroid']
            xDist = abs(centroidOne[0] - centroidTwo[0])
            yDist = abs(centroidOne[1] - centroidTwo[1])
            newVelocityX = (B*(xDist / samplingInterval)) + ((1-B)*blobs[blobOne]['velocityX'])
            newVelocityY = (B*(yDist / samplingInterval)) + ((1-B)*blobs[blobOne]['velocityY'])
            blobs[blobTwo]['velocityX'] = newVelocityX
            blobs[blobTwo]['velocityY'] = newVelocityY

    return blobs



cap = cv2.VideoCapture('ZebraFIshOneWell.mov')

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
samplingInterval= 1 / fps
ret,frame1 = cap.read()
grayFrameOne=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
backgroundImage=cv2.imread('zebraFishBackgroundImage.png')
grayBackground=cv2.cvtColor(backgroundImage,cv2.COLOR_BGR2GRAY)

differenceImageOne = cv2.absdiff(grayFrameOne,grayBackground)


blurOne =cv2.GaussianBlur(differenceImageOne,(5,5),0)

_,threshOne=cv2.threshold(blurOne,25,255,cv2.THRESH_BINARY)
erodedImageOne=cv2.erode(threshOne,None,1)

dilatedImageOne=cv2.dilate(erodedImageOne,None,1)



(numLabelsOne, labelsOne, statsOne, centroidsOne) = cv2.connectedComponentsWithStats(dilatedImageOne,4,cv2.CV_32S)

blobPropertiesOne=[]


k=1
for i in range(1,numLabelsOne):
    area=statsOne[i, cv2.CC_STAT_AREA]
    x = statsOne[i, cv2.CC_STAT_LEFT]
    y = statsOne[i, cv2.CC_STAT_TOP]
    w=statsOne[i, cv2.CC_STAT_WIDTH]
    h=statsOne[i, cv2.CC_STAT_HEIGHT]
    boundingBoxArea=w*h
    density=area/boundingBoxArea
    velocity=0
    Property={
        'area' : area,
        'boxArea' : boundingBoxArea,
        'density' : density,
        'velocityX' : 0,
        'velocityY' : 0,
        'boxWidth': w,
        'boxHeight' : h,
        'centroid' : centroidsOne[i],
        'frameNumber' : 1,
        'topLeft': Point(x,y),

    }
    blobPropertiesOne.append(Property)




k=1
while cap.isOpened():
    ret,frame2=cap.read()
    grayFrameTwo=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    differenceImageTwo = cv2.absdiff(grayFrameTwo,grayBackground)
    blurTwo =cv2.GaussianBlur(differenceImageTwo,(5,5),0)
    _,threshTwo=cv2.threshold(blurTwo,25,255,cv2.THRESH_BINARY)
    erodedImageTwo=cv2.erode(threshTwo,None,1)
    dilatedImageTwo=cv2.dilate(erodedImageTwo,None,1)

    (numLabelsTwo, labelsTwo, statsTwo, centroidsTwo) = cv2.connectedComponentsWithStats(dilatedImageTwo,4,cv2.CV_32S)


    blobPropertiesTwo=[]
    for i in range(1,numLabelsTwo):
        area=statsTwo[i, cv2.CC_STAT_AREA]
        x = statsTwo[i, cv2.CC_STAT_LEFT]
        y = statsTwo[i, cv2.CC_STAT_TOP]
        w=statsTwo[i, cv2.CC_STAT_WIDTH]
        h=statsTwo[i, cv2.CC_STAT_HEIGHT]
        boundingBoxArea=w*h
        density=area/boundingBoxArea
        velocity=0
        Property={
            'area' : area,
            'boxArea' : boundingBoxArea,
            'density' : density,
            'velocityX' : 0,
            'velocityY' : 0,
            'boxWidth': w,
            'boxHeight' : h,
            'centroid' : centroidsTwo[i],
            'frameNumber' : 2,
            'topLeft': Point(x,y),
        }
        blobPropertiesTwo.append(Property)

    """
    for blob in blobPropertiesTwo:
        boundingBoxTopLeft=blob['topLeft']
        boundingBoxWidth=blob['boxWidth']
        boundingBoxHeight=blob['boxHeight']
        cv2.rectangle(frame2,(boundingBoxTopLeft.x,boundingBoxTopLeft.y),(boundingBoxTopLeft.x+boundingBoxWidth,boundingBoxTopLeft.y+boundingBoxHeight),(0,255,0))
    
    cv2.imshow('frame2',frame2)
    cv2.waitKey(0)

    """

   
    Graph,blobs,Edges=FormGraph(blobPropertiesOne,blobPropertiesTwo)
    print(len(Edges))
    buildGraph=BuildGraph()
    buildGraph.RemoveEdges(0,Graph,Edges,blobs) 
    minGraph=buildGraph.minGraph
    newEdges=GetEdges(minGraph)
    newBlobs = ComputeVelocities(blobs,minGraph,newEdges,samplingInterval)
    for edge in newEdges:
        blobTwo=blobs[edge[1]]
        boundingBoxTopLeft=blobTwo['topLeft']
        boundingBoxWidth=blobTwo['boxWidth']
        boundingBoxHeight=blobTwo['boxHeight']
        cv2.rectangle(frame2,(boundingBoxTopLeft.x,boundingBoxTopLeft.y),(boundingBoxTopLeft.x+boundingBoxWidth,boundingBoxTopLeft.y+boundingBoxHeight),(0,255,0))
        cv2.putText(frame2,str(edge[1]),(boundingBoxTopLeft.x,boundingBoxTopLeft.y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('frame2',frame2)
    cv2.waitKey(30)
    """
    components=ConnectedComponents(Graph)
    maxLen=0
    for i in range(len(components)):
        maxLen=max(maxLen,countEdges(components[i],Graph))
    print(maxLen)
    print(len(Edges))
    optimumGraph=RemoveEdges(0,Graph,Edges,float("inf"),blobs,[])
    print(optimumGraph)
 
    l=8852
    corner=blobs[l]['topLeft']
    cornerWidth=blobs[l]['boxWidth']
    cornerHeight=blobs[l]['boxWidth']
    cv2.rectangle(frame1,(corner.x,corner.y),(corner.x+cornerWidth,corner.y+cornerHeight),(255,0,0))
    for blob in Graph[l]:
        boundingBoxTopLeft=blobs[blob]['topLeft']
        boundingBoxWidth=blobs[blob]['boxWidth']
        boundingBoxHeight=blobs[blob]['boxWidth']
        cv2.rectangle(frame2,(boundingBoxTopLeft.x,boundingBoxTopLeft.y),(boundingBoxTopLeft.x+boundingBoxWidth,boundingBoxTopLeft.y+boundingBoxHeight),(0,255,0))
    
    cv2.imshow('frame1',frame1)
    cv2.imshow('frame2',frame2)
    cv2.waitKey(0)



    blobPropertiesOne=[]
    for i in range(len(blobPropertiesTwo)):
        blobPropertiesTwo[i]['frameNumber']=1
    """
    
    blobPropertiesOne=[]
    for blob in newBlobs:
        if blob['frameNumber'] == 2:
            blob['frameNumber'] = 1
            blobPropertiesOne.append(blob)
    
    print(blobPropertiesOne)


"""
while cap.isOpened():

    diff=cv2.absdiff(frame1,frame2)

    gray=cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur =cv2.GaussianBlur(gray,(5,5),0)
    _,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)

    cv2.imshow('Threshold',thresh)

    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour) 
        area=cv2.contourArea(contour)
        print("Contour Area: "+str(area))
        if cv2.contourArea(contour) < 50 :
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h), (0,255,0),2)
        cv2.putText(frame1,"Status : {}".format('Movement'), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    #cv2.drawContours(frame1,contours, -1, (0,255,0), 2)

    cv2.imshow("feed",frame1)

    frame1 = frame2

    ret, frame2 = cap.read()
    if not ret:
        break

    if cv2.waitKey(40) == 27:
        break


"""
cv2.destroyAllWindows()
cap.release()