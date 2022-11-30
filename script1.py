from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.insert(1, '/home/nikola/test/osv')
from vaja_1.script2 import loadImage, displayImage

if __name__ == '__main__':
    orig_size = [256,256]

    I = loadImage('/home/nikola/test/osv/vaja_7/cameraman-256x256-08bit.raw',orig_size,np.uint8)
    displayImage(I,'Orginalna slika',iGridX=[0,255],iGridY=[0,255])

def spatialFiltering ( iType , iImage , iFilter , iStatFunc = None ,iMorphOp = None ) :
    N,M=iFilter.shape
    # n so vrstice, m so stolpce
    n = int((N-1)/2)
    m = int((M-1)/2)

    iImage=changeSpatialDomain(iType='enlarge',iImage=iImage,iX=m,iY=n)

    Y,X = iImage.shape
    oImage = np.zeros((Y,X),dtype=float)

    for y in range(n,Y-n):
        for x in range (m,X-m):
            patch = iImage[y - n : y + n + 1 , x - m : x + m +1]

            if iType =='kernel':
                oImage[y,x]= (patch *iFilter).sum()
            elif iType == 'statistical':
                oImage[y,x]=iStatFunc(patch)
            elif iType == 'morphological':
                R=patch[iFilter !=0]
                if iMorphOp == 'erosion':
                    oImage[y,x]=R.min()
                elif iMorphOp == 'dilation':
                    oImage[y,x]=R.max()
    
    oImage = changeSpatialDomain(iType='reduce',iImage=oImage,iX=m,iY=m)
    return oImage

def changeSpatialDomain ( iType , iImage , iX , iY , iMode = None , iBgr =0) :
    Y,X=iImage.shape

    if iType == 'enlarge':
        if iMode is None:
            oImage = np.zeros((Y+2*iY,X+2*iX),dtype=float)
            oImage[iY:Y+iY,iX:X+iX]=iImage
        if iMode == 'constant':
            
            oImage = np.ones((Y+2*iY,X+2*iX),dtype=float)*iBgr
            oImage[iY:Y+iY,iX:X+iX]=iImage
        if iMode == 'extrapolation':
            oImage = np.zeros((Y+2*iY,X+2*iX),dtype=float)
          
            h_gor = iImage[0,:]
            h_dol = iImage[iImage.shape[0]-1,:]
            v_levo = iImage[:,0]
            v_desno = iImage[:,iImage.shape[1]-1]

            oImage[0:iY,0:iX]=h_gor[0]
            oImage[iY+iImage.shape[0]:oImage.shape[0],0:iX]=h_dol[0]

            oImage[0:iY,iX+iImage.shape[1]:oImage.shape[1]]=v_desno[0]
            oImage[iY+iImage.shape[0]:oImage.shape[0],iX+iImage.shape[1]:oImage.shape[1]]=v_desno[iImage.shape[1]-1]

            for i in range(iY):
                oImage[i,iX:iX+iImage.shape[1]]=h_gor
                oImage[i+iImage.shape[0]+iY,iX:iX+iImage.shape[1]]=h_dol
            for i in range(iX):
                oImage[iY:iY+iImage.shape[0],i]=v_levo
                oImage[iY:iY+iImage.shape[0],i+iX+iImage.shape[1]]=v_desno
              
            oImage[iY:iY+iImage.shape[0],iX:iX+iImage.shape[1]]=iImage[:,:]
        
        if iMode == 'reflection':

            oImage = np.zeros((Y+2*iY,X+2*iX),dtype=float)
            oImage[iY:iY+iImage.shape[0],iX:iX+iImage.shape[1]]=iImage[:,:]
            
            
            ratio_y = iY/iImage.shape[0]
            reminder_y = ratio_y-np.floor(ratio_y)
            flip_y=-1
            counter_y =np.floor(ratio_y)
            for i in range(int(counter_y)):

                    sp_x=iX #spodnja meja x
                    zg_x = sp_x + iImage.shape[1] #zgornja meja x

                    zg_gor_y = iY - i*iImage.shape[0] #zgornja meja gor y
                    sp_gor_y = zg_gor_y - iImage.shape[0] #spodnja meja gor y

                    zg_dol_y = iY + 2*iImage.shape[0] + i*iImage.shape[0] #zgornja meja desno x
                    sp_dol_y = zg_dol_y - iImage.shape[0] #spodnja meja desno x

                    oImage[sp_gor_y : zg_gor_y, sp_x : zg_x]=iImage[0:iImage.shape[0],:][::flip_y,:]  #gor
                    oImage[sp_dol_y : zg_dol_y, sp_x : zg_x]=iImage[0:iImage.shape[0],:][::flip_y,:]  #dol
                    flip_y=flip_y*(-1)
        
            if counter_y%2==0:
                oImage[0 : int(iY-counter_y*iImage.shape[0]), sp_x : zg_x]= iImage[0:int(iImage.shape[0]*reminder_y),:][::flip_y,:] #gor
                oImage[int(iY+counter_y*iImage.shape[0]+iImage.shape[0]):oImage.shape[0],sp_x : zg_x]= iImage[iImage.shape[0]-int(iImage.shape[0]*reminder_y):iImage.shape[0],:][::flip_y,:] #dol          
            else:
                oImage[0:int(iY-counter_y*iImage.shape[0]), sp_x : zg_x]= iImage[iImage.shape[0]-int(iImage.shape[0]*reminder_y):iImage.shape[0],:][::flip_y,:]   #gor  
                oImage[int(iY+counter_y*iImage.shape[0]+iImage.shape[1]):oImage.shape[0],sp_x : zg_x]= iImage[0:int(iImage.shape[0]*reminder_y),:][::flip_y,:]   #dol   


            ratio_x = iX/iImage.shape[1]
            reminder_x = ratio_x-np.floor(ratio_x)
            flip_x=-1
            counter_x =np.floor(ratio_x)
            for i in range(int(counter_x)):
                    sp_y=iY #spodnja meja y
                    zg_y = sp_y + iImage.shape[0] #zgornja meja y

                    zg_levo_x = iX - i*iImage.shape[1] #zgornja meja levo x
                    sp_levo_x = zg_levo_x - iImage.shape[1] #spodnja meja levo x

                    zg_desno_x = iX + 2*iImage.shape[1] + i*iImage.shape[1] #zgornja meja desno x
                    sp_desno_x = zg_desno_x - iImage.shape[1] #spodnja meja desno x
                    
                    oImage[sp_y : zg_y,sp_levo_x:zg_levo_x]=iImage[:,0:iImage.shape[1]][:,::flip_x]  #levo
                    oImage[sp_y : zg_y,sp_desno_x:zg_desno_x]=iImage[:,0:iImage.shape[1]][:,::flip_x]  #desno

                    flip_x=flip_x*(-1)
        
            if counter_x%2==0:
                oImage[iY:iY+iImage.shape[0],0:int(iX-counter_x*iImage.shape[1])]= iImage[:,0:int(iImage.shape[1]*reminder_x)][:,::flip_x] #levo
                oImage[iY:iY+iImage.shape[0],int(iX+counter_x*iImage.shape[1]+iImage.shape[1]):oImage.shape[1]]= iImage[:,iImage.shape[1]-int(iImage.shape[1]*reminder_x):iImage.shape[1]][:,::flip_x] #desno          
            else:
                oImage[iY:iY+iImage.shape[0],0:int(iX-counter_x*iImage.shape[1])]= iImage[:,iImage.shape[1]-int(iImage.shape[1]*reminder_x):iImage.shape[1]][:,::flip_x]   #levo
                oImage[iY:iY+iImage.shape[0],int(iX+counter_x*iImage.shape[1]+iImage.shape[0]):oImage.shape[1]]= iImage[:,0:int(iImage.shape[1]*reminder_x)][:,::flip_x]   #desno 

            
          
    
           
    elif iType == 'reduce':
        if iMode is None:
            oImage = np.copy(iImage[iY:Y-iY,iX:X-iX])

            
    return oImage


def weightedAverageFilter ( iM, iN ,iValue ) :

    if(iM%2 !=0 and iN%2 !=0):
        oFilter=np.ones((iM,iN),dtype=int)
        place_holder=[None]*iM
        for y in range (int((iN+1)/2)):
            for x in range (int((iM+1)/2)):
                oFilter[x,y]=iValue**x *iValue**y
                place_holder [:]= oFilter[::-1,y]
                oFilter[int((iM+1)/2):iM,y] = place_holder[int((iM+1)/2):iM]
            place_holder [:]= oFilter[::-1,y]
            oFilter[:,iN-1-y] = place_holder[:] 
        #oFilter = oFilter/oFilter.sum()      
    else:
        print("Velikosti filtra morajo biti lihe številke")

    return oFilter




if __name__ == '__main__':

    print("NALOGA 1")
    print("-----------------------------------------------------------------------------------------------")
    K = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])
    SE = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
    kI=spatialFiltering(iType='kernel',iImage=I,iFilter=K)
    displayImage(kI,"Slika po filtriranju z jedro",iGridX=(-0.5,orig_size[0]-0.5),iGridY=[-0.5,orig_size[1]-0.5])

    sI=spatialFiltering(iType='statistical',iImage=I,iFilter=np.zeros([3,3]),iStatFunc=np.median)
    displayImage(sI,"Slika po statisticnem filtriranju")

    mI=spatialFiltering(iType='morphological',iImage=I,iFilter=SE,iMorphOp='erosion')
    displayImage(mI,"Slika po morfoloskim filtriranju")


    print("NALOGA 2")
    print("-----------------------------------------------------------------------------------------------")
    print(weightedAverageFilter(7,5,2))

    print("NALOGA 3")
    print("-----------------------------------------------------------------------------------------------")
    gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    xI = spatialFiltering(iType='kernel',iImage=I,iFilter=gx)
    displayImage(xI,"xI")
    yI = spatialFiltering(iType='kernel',iImage=I,iFilter=gy)
    displayImage(yI,"yI")
    GI = np.sqrt(xI**2+yI**2)
    GI=GI/np.max(GI)*255
    displayImage(GI,"Amplitudni odziv")
    xI=xI+0.0000001
    yI=yI+0.0000001
    aI=np.arctan2(xI,yI)
    aI=aI/np.max(aI)*255
    displayImage(aI,"Fazni odziv")

    print("NALOGA 4")
    print("-----------------------------------------------------------------------------------------------")
    gauss = np.array([[0.01,0.08,0.01],[0.08,0.64,0.08],[0.01,0.08,0.01]])
    c=2
    FI = spatialFiltering(iType='kernel',iImage=I,iFilter=gauss)
    mI = I - FI
    gI = I + c*mI
    displayImage(gI,"Ostrenje z maskiranjem neostrih področjih")

    print("NALOGA 5")
    print("-----------------------------------------------------------------------------------------------")
    iI = changeSpatialDomain('enlarge',iImage=I,iX=128,iY=384,iMode='constant',iBgr=127)
    displayImage(iI,"const")
    iI = changeSpatialDomain('enlarge',iImage=I,iX=128,iY=384,iMode='extrapolation',iBgr=127)
    displayImage(iI,"extr")

    iI = changeSpatialDomain('enlarge',iImage=I,iX=384,iY=384,iMode='reflection',iBgr=127)
    displayImage(iI,"ref")

