#!/usr/bin/env python 
import os
if __name__ == '__main__':
    os.environ["MPLBACKEND"] = "Agg" 

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import colormaps 
import numpy as np
import timeit
import pickle
import argparse
import multiprocess as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')    

colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

def plotfaces(R, ax, cmap, Rmax, Ns, Ls):
    Y,X=np.meshgrid(np.arange(Ns[0])/Ns[0]*Ls[0],np.arange(Ns[1])/Ns[1]*Ls[1])
    Z=Ls[2]*np.ones(X.shape)
    C=cmap(R[5]/Rmax)
    ax.plot_surface(X,Y,Z,rstride=1, cstride=1, facecolors=C, shade=False)
    
    X,Z=np.meshgrid(np.arange(Ns[0])/Ns[0]*Ls[0],np.arange(Ns[2])/Ns[2]*Ls[2])
    Y=0*np.ones(X.shape)
    C=cmap(R[1].T/Rmax)
    ax.plot_surface(X,Y,Z,rstride=1, cstride=1, facecolors=C, shade=False)
    
    Z,Y=np.meshgrid(np.arange(Ns[1])/Ns[1]*Ls[1],np.arange(Ns[2])/Ns[2]*Ls[2])
    X=Ls[0]*np.ones(X.shape)
    C=cmap(R[3]/Rmax)
    ax.plot_surface(X,Y,Z,rstride=1, cstride=1, facecolors=C, shade=False)
    
    ax.plot([Ls[0],Ls[0]],[0,0],[0,Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,Ls[0]],[0,0],[Ls[2],Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([Ls[0],Ls[0]],[0,Ls[1]],[Ls[2],Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,Ls[0]],[0,0],[0,0], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([Ls[0],Ls[0]],[0,Ls[1]],[0,0], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,0],[0,0],[0,Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([Ls[0],Ls[0]],[Ls[1],Ls[1]],[0,Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,Ls[0]],[Ls[1],Ls[1]],[Ls[2],Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,0],[0,Ls[1]],[Ls[2],Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    
    ax.set_xlim(0,Ls[0])
    ax.set_ylim(0,Ls[1])
    ax.set_zlim(0,Ls[2])
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    
def plotlines(lines, ids, ax, Ns, Ls):
    for i in range(len(lines)):
        c=colors[np.mod(ids[i],len(colors))]
        for line in lines[i]:
            if np.linalg.norm(line[0]-line[1],ord=np.inf)<np.min(Ls)/2:
                alpha=0.25*np.linalg.norm(([0,Ls[1],0]-line[0])/Ls,ord=np.inf)
                ax.plot(*line.T,lw=1.5, marker=None, c=c, alpha=alpha)
            else:
                if (line[0][0]==0 or line[0][1]==Ls[1] or line[0][2]==0) and not (line[1][0]==0 or line[1][1]==Ls[1] or line[1][2]==0):
                    ax.scatter(*line[0],s=1,color=c,alpha=0.25)
                if (line[0][0]==Ls[0] or line[0][1]==0 or line[0][2]==Ls[2]) and not (line[1][0]==Ls[0] or line[1][1]==0 or line[1][2]==Ls[2]):
                    ax.scatter(*line[0],s=1,color=c,alpha=1.0)
                if (line[1][0]==0 or line[1][1]==Ls[1] or line[1][2]==0) and not (line[0][0]==0 or line[0][1]==Ls[1] or line[0][2]==0):
                    ax.scatter(*line[1],s=1,color=c,alpha=0.25)
                if (line[1][0]==Ls[0] or line[1][1]==0 or line[1][2]==Ls[2]) and not (line[0][0]==Ls[0] or line[0][1]==0 or line[0][2]==Ls[2]):
                    ax.scatter(*line[1],s=1,color=c,alpha=1.0)
    
    ax.plot([Ls[0],Ls[0]],[0,0],[0,Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,Ls[0]],[0,0],[Ls[2],Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([Ls[0],Ls[0]],[0,Ls[1]],[Ls[2],Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,Ls[0]],[0,0],[0,0], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([Ls[0],Ls[0]],[0,Ls[1]],[0,0], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,0],[0,0],[0,Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([Ls[0],Ls[0]],[Ls[1],Ls[1]],[0,Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,Ls[0]],[Ls[1],Ls[1]],[Ls[2],Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    ax.plot([0,0],[0,Ls[1]],[Ls[2],Ls[2]], lw=1, c='black', alpha=1, zorder=10)
    
    ax.set_xlim(0,Ls[0])
    ax.set_ylim(0,Ls[1])
    ax.set_zlim(0,Ls[1])
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

def makeplot(n, Ns, Ls, faces=None, lines=None, cids=None, Rmax=1, filebase=None, thmax=2*np.pi, show=False, save=True):
    if save and os.path.exists(filebase+'animation/%04d.png'%(n)):
        return

    subplots=0
    if faces is not None:
        subplots=subplots+2
    if lines is not None:
        subplots=subplots+1
        
    fig = plt.figure(figsize=(4*subplots,4), layout='constrained')

    l=1
    if faces is not None:
        ax = fig.add_subplot(1,subplots,l, projection='3d')
        tmp_planes = ax.zaxis._PLANES
        ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3], tmp_planes[0], tmp_planes[1], tmp_planes[4], tmp_planes[5])
        R=[np.linalg.norm(face,axis=0) for face in faces[n]]

        plotfaces(R, ax, plt.get_cmap('coolwarm'), Rmax, Ns, Ls)
        fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=Rmax), cmap=plt.get_cmap('coolwarm')),ax=ax,location='right', shrink=0.5, pad=-0.05)
        ax.set_box_aspect(aspect=None, zoom=0.8) 
        l=l+1
    
        ax = fig.add_subplot(1,subplots,l, projection='3d')
        tmp_planes = ax.zaxis._PLANES
        ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3], tmp_planes[0], tmp_planes[1], tmp_planes[4], tmp_planes[5])
        th=[np.atan2(face[1],face[0])+np.pi for face in faces[n]]

        plotfaces(th, ax, plt.get_cmap('twilight'), thmax, Ns, Ls)
        fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=thmax), cmap=plt.get_cmap('twilight')),ax=ax,location='right', shrink=0.5, pad=-0.05)
        ax.set_box_aspect(aspect=None, zoom=0.8) 
        l=l+1

    if lines is not None:
        ax = fig.add_subplot(1,subplots,l, projection='3d')
        tmp_planes = ax.zaxis._PLANES
        ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3], tmp_planes[0], tmp_planes[1], tmp_planes[4], tmp_planes[5])
        plotlines(lines[n], cids[n], ax, Ns, Ls)
        #invisible colorbar to scale the size consistently
        cbar=fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=plt.get_cmap('gist_rainbow')),ax=ax,location='right', shrink=0.5, pad=-0.05,alpha=0)
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor('white')
        cbar.ax.tick_params(color='white', labelcolor='white')
        ax.set_box_aspect(aspect=None, zoom=0.8) 

    if save:
        print(filebase+'animation/%04d.png'%(n),flush=True)
        plt.savefig(filebase+'animation/%04d.png'%(n),bbox_inches="tight", pad_inches=0.1)
    if show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser(description='3DCGLE defect lines.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
    parser.add_argument("--Ns", type=int, nargs=3, default=[128,128,128], dest='Ns', help='Grid points in each dimension')
    parser.add_argument("--Ls", type=float, nargs=3, default=[200.0,200.0,200.0], dest='Ls', help='Grid points in each dimension')
    parser.add_argument("--threads", type=int, default=16, dest='threads', help='Number of threads to use')
    parser.add_argument("--rm", type=int, default=0, dest='rm', help='Remove pngs')
    args = parser.parse_args()

    filebase=args.filebase
    Ns=np.array(args.Ns)
    Ls=np.array(args.Ls)
    threads=args.threads
    
    if not os.path.exists(filebase+'animation'):
        os.mkdir(filebase+'animation')
    elif args.rm:
        os.system('rm %sanimation/*.png'%(filebase))
    
    file=open('%slines.dat'%filebase, 'rb')
    lines=pickle.load(file)
    file.close()
    
    file=open('%slineids.dat'%filebase, 'rb')
    ids,cids,edges=pickle.load(file)
    file.close()
    nt=len(lines)

    file=open("%sfaces.dat"%(filebase),"rb")
    faces=pickle.load(file)
    file.close()

    nt=len(lines)
    Rmax=np.max(faces)
    print(Rmax, nt, filebase)

    start=timeit.default_timer()
    pool=mp.Pool(threads,maxtasksperchild=1) 
    pool.starmap(makeplot, [(n, Ns, Ls, faces, lines, cids, Rmax, filebase) for n in range(nt)])
    pool.close()
    os.system('ffmpeg -y -r 30 -i %sanimation/'%(filebase)+'%04d.png'+ ' -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p" %sanimation.mp4'%(filebase))
    stop=timeit.default_timer()
    print("plot time: ", stop-start)

    if args.rm:
        os.system('rm -r %sanimation'%(filebase))