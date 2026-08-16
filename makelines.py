#!/usr/bin/env python 
import os
import numpy as np
import timeit
import subprocess
import struct
from scipy.sparse import coo_array
import networkx as nx
import pickle
import argparse
import sys
import multiprocess as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')

def wloop(th, axis1=0, axis2=1):
    dth1=np.mod(np.roll(th,-1,axis=axis1)-th+np.pi,2*np.pi)-np.pi
    dth2=np.mod(np.roll(np.roll(th,-1,axis=axis1),-1,axis=axis2)-np.roll(th,-1,axis=axis1)+np.pi,2*np.pi)-np.pi
    dth3=np.mod(np.roll(th,-1,axis=axis2)-np.roll(np.roll(th,-1,axis=axis1),-1,axis=axis2)+np.pi,2*np.pi)-np.pi
    dth4=np.mod(th-np.roll(th,-1,axis=axis2)+np.pi,2*np.pi)-np.pi
    return np.round((dth1+dth2+dth3+dth4)/(2*np.pi))

def findlines(th,Ns,Ls,chunk):
    #each cube sharing a face with a line passing through it
    lines01=coo_array(wloop(th, axis1=0, axis2=1)).coords
    lines12=coo_array(wloop(th, axis1=1, axis2=2)).coords
    lines20=coo_array(wloop(th, axis1=2, axis2=0)).coords
    lines012=lines01+np.array([0,0,-1])[:,np.newaxis]
    lines012[2]=np.mod(lines012[2]+Ns[2],Ns[2])
    lines120=lines12+np.array([-1,0,0])[:,np.newaxis]
    lines120[0]=np.mod(lines120[0]+Ns[0],Ns[0])
    lines201=lines20+np.array([0,-1,0])[:,np.newaxis]
    lines201[1]=np.mod(lines201[1]+Ns[1],Ns[1])

    lines=np.concatenate([lines01,lines012,lines12,lines120,lines20,lines201],axis=1)
    raveled=np.apply_along_axis(lambda row: np.ravel_multi_index(row,Ns), axis=0, arr=lines)
    lines=np.array(np.unravel_index(np.unique(raveled),shape=Ns))

    nnz=lines.shape[1]
    #print(nnz)
    adj=coo_array((nnz,nnz))
    for n in range(nnz//chunk+1):
        #print(n,nnz//chunk+1,end='\r')
        diffs=(lines[:,n*chunk:(n+1)*chunk,np.newaxis]-lines[:,np.newaxis,:])
        for axis in range(3):
            diffs[axis]=(np.mod(diffs[axis]+Ns[axis]//2,Ns[axis])-Ns[axis]//2)
        rows,cols=np.where(np.linalg.norm(diffs,ord=1,axis=0).astype(int)==1)
        coords=np.array([rows+n*chunk,cols])
        nnzA=len(rows)
        adj=adj+coo_array((np.ones(nnzA),coords),shape=(nnz,nnz))
        
    G=nx.from_numpy_array(adj)
    lines=[(lines[:,list(G.subgraph(c).edges)]).transpose((1,2,0))*Ls/(Ns-1) for c in nx.connected_components(G)]
    return lines

def savelines(n,filename,Ns,Ls,n0,chunk):
    file=open(filename,'rb')
    file.seek((n0+n)*8*2*Ns[0]*Ns[1]*Ns[2])
    dat=file.read(8*2*Ns[0]*Ns[1]*Ns[2])
    ys=np.array(struct.unpack('%id'%(2*Ns[0]*Ns[1]*Ns[2]),dat)).reshape([2]+Ns.tolist())
    file.close()
    th=np.atan2(ys[1],ys[0])+np.pi
    lines=findlines(th,Ns,Ls,chunk)
    return lines

def correlate(n,lines,Ns,Ls,chunk):
    diffs=np.zeros((len(lines[n]),len(lines[n-1])),dtype=int)
    for i in range(len(lines[n])):
        for j in range(len(lines[n-1])):
            nnz1=len(lines[n][i])
            nnz2=len(lines[n-1][j])
            # dist=np.zeros((nnz1,nnz2,2))
            # for m in range(nnz1//chunk+1):
            #     dist[m*chunk:(m+1)*chunk]=np.linalg.norm(lines[n][i][m*chunk:(m+1)*chunk,np.newaxis]/(Ls/(Ns-1))-lines[n-1][j][np.newaxis,:]/(Ls/(Ns-1)),ord=1,axis=-1)
            # diffs[i,j]=np.round(np.min(dist))
            diffs[i,j]=np.max(Ns)
            for m in range(nnz1//chunk+1):
                dist=np.linalg.norm(lines[n][i][m*chunk:(m+1)*chunk,np.newaxis]/(Ls/(Ns-1))-lines[n-1][j][np.newaxis,:]/(Ls/(Ns-1)),ord=1,axis=-1)
                if len(dist)>0:
                    diffs[i,j]=np.min([np.round(np.min(dist)),diffs[i,j]])
    return diffs

def loadfaces(n,filebase,Ns,n0):
    file=open(filebase+'states.dat','rb')
    file.seek((n0+n)*8*2*Ns[0]*Ns[1]*Ns[2])
    dat=file.read(8*2*Ns[0]*Ns[1]*Ns[2])
    ys=np.array(struct.unpack('%id'%(2*Ns[0]*Ns[1]*Ns[2]),dat)).reshape([2]+Ns.tolist())
    file.close()
    return [ys[:,0,:,:],ys[:,:,0,:],ys[:,:,:,0],ys[:,-1,:,:],ys[:,:,-1,:],ys[:,:,:,-1]]




if __name__ == "__main__":


    loadstr=''
    try:
        loadstr='module load %s &&'%(subprocess.check_output("module avail | grep cuda/", shell=True, text=True).split()[0])
    except:
        pass

    print(*sys.argv)

    #Command line arguments
    parser = argparse.ArgumentParser(description='3DCGLE defect lines.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
    parser.add_argument("--Ns", type=int, nargs=3, default=[128,128,128], dest='Ns', help='Grid points in each dimension')
    parser.add_argument("--Ls", type=float, nargs=3, default=[50.0,50.0,50.0], dest='Ls', help='Grid points in each dimension')
    parser.add_argument("--b", type=float, default=0.5, dest='b', help='CGLE b parameter')
    parser.add_argument("--c", type=float, default=2.0, dest='c', help='CGLE c parameter')
    parser.add_argument("--seed", type=int, default=100, dest='seed', help='Random seed')
    parser.add_argument("--T0", type=float, default=0.1, dest='T0', help='Fixed step burn in time')
    parser.add_argument("--dt0", type=float, default=1E-3, dest='dt0', help='Fixed step burn in time step, between 0 and T0')
    parser.add_argument("--T", type=float, default=1000, dest='T', help='Total integration time')
    parser.add_argument("--dt", type=float, default=10.0, dest='dt', help='Final integration output time step, between T1 and T')
    parser.add_argument("--T1", type=float, default=900, dest='T1', help='Time to start tracking lines')
    parser.add_argument("--dt1", type=float, default=0.1, dest='dt1', help='Initial integration output time step, between T0 and T1')
    parser.add_argument("--lines", type=int, default=1, dest='dolines', help='Flag to track lines')
    parser.add_argument("--states", type=int, default=1, dest='dostates', help='Flag to generate states')
    parser.add_argument("--threads", type=int, default=4, dest='threads', help='Number of threads to use')
    parser.add_argument("--chunk", type=int, default=1024, dest='chunk', help='Chunk size')
    parser.add_argument("--rm", type=int, default=0, dest='rm', help='Remove states')
    args = parser.parse_args()

    Ns=np.array(args.Ns)
    Ls=np.array(args.Ls)
    b=args.b
    c=args.c
    seed=args.seed
    filebase=args.filebase
    T0=args.T0
    dt0=args.dt0
    T=args.T
    dt=args.dt
    T1=args.T1
    dt1=args.dt1
    threads=args.threads
    chunk=args.chunk
    rm=args.rm
    
    n0=1+int(T0/dt0)+int(T1/dt1)
    nt=int((T-T1)/dt)

    if args.dostates:

        couplings=['1.0,0,0,1', '1.0,0,8,1', '1.0,0,16,1', '1.0,0,24,1', '-%f,0,9,1'%(b), \
                '-%f,0,17,1'%(b), '-%f,0,25,1'%(b), '-1.0,0,0,3', '-1.0,0,0,1,1,2', '1.0,1,1,1', \
                '1.0,1,9,1', '1.0,1,17,1', '1.0,1,25,1', '%f,1,8,1'%(b),'%f,1,16,1'%(b), '%f,1,24,1'%(b), \
                '-1.0,1,0,2,1,1', '-1.0,1,1,3', '%f,1,0,3'%(c), '%f,1,0,1,1,2'%(c), \
                '-%f,0,0,2,1,1'%(c),'-%f,0,1,3'%(c)]

        file=open('%scoupling.dat'%(filebase),'w')
        for line in couplings:
            print(line,file=file)
        file.close()


        cmd='%s ./rdcu -s 1 -t %f -d %f -N %i,%i,%i -L %f,%f,%f -s %i -n 2 -D1 -F %s'%(loadstr,T0,dt0,Ns[0],Ns[1],Ns[2],Ls[0],Ls[1],Ls[2],seed,filebase)
        print(cmd,flush=True)
        os.system(cmd)
        cmd='%s ./rdcu -s 1 -t %f -d %f -N %i,%i,%i -L %f,%f,%f -s %i -n 2 -D1 -R  %s'%(loadstr,T1,dt1,Ns[0],Ns[1],Ns[2],Ls[0],Ls[1],Ls[2],seed,filebase)
        print(cmd,flush=True)
        os.system(cmd)
        cmd='%s ./rdcu -s 1 -t %f -d %f -N %i,%i,%i -L %f,%f,%f -s %i -n 2 -D1 -R  %s'%(loadstr,T,dt,Ns[0],Ns[1],Ns[2],Ls[0],Ls[1],Ls[2],seed,filebase)
        print(cmd,flush=True)
        os.system(cmd)

    if args.dolines:
        if not os.path.exists(filebase+'lines.dat'):
            start=timeit.default_timer()
            pool=mp.Pool(threads,maxtasksperchild=1) 
            lines=pool.starmap(savelines,[(n,'%sstates.dat',filebase,Ns,Ls,n0,chunk) for n in range(nt)])
            pool.close()
            stop=timeit.default_timer()
            print("lines time: ", stop-start,flush=True)

            file=open(filebase+'lines.dat','wb')
            pickle.dump(lines,file)
            file.close()
        else:
            file=open(filebase+'lines.dat','rb')
            lines=pickle.load(file)
            file.close()

        start=timeit.default_timer()
        pool=mp.Pool(threads,maxtasksperchild=1) 
        results=pool.starmap(correlate, [(n,lines,Ns,Ls,chunk) for n in range(1,nt)])
        pool.close()
        stop=timeit.default_timer()
        print("correlate time: ", stop-start,flush=True)


        start=timeit.default_timer()

        lengths=[]
        for line in lines:
            length=[]
            for l in line:
                length=length+[len(l)]
            lengths=lengths+[length]

        th=5 #threshold distance to determine if lines at different timesteps are related
        parentinds=[np.arange(len(lines[0])).reshape((-1,1)).tolist()]
        for n in range(nt-1):
            ps=[[]]*results[n].shape[0]
            for i in range(results[n].shape[0]):
                mdist=np.min(results[n][i])
                if mdist<=th:
                    ps[i]=np.where(results[n][i]==mdist)[0].tolist()
            parentinds=parentinds+[ps] 

        siblinginds=[]
        for pinds in parentinds:
            sinds=[]
            pstrs=["_".join(np.array(p,dtype=str)) for p in pinds]

            for pstr in pstrs:
                s=np.where(pstr==np.array(pstrs))[0]
                sinds=sinds+[s.tolist()]
            siblinginds=siblinginds+[sinds]


        #find the id and color of the lines
        #if one or more lines spawn children, the longest child inherits the longest parent's color
        ids=[np.arange(len(lines[0]))]
        cids=[np.arange(len(lines[0]))]
        nextid=len(lines[0])
        nextcid=len(lines[0])
        edges=[]
        for n in np.arange(1,nt):
            i=np.zeros(len(parentinds[n]),dtype=int)
            ci=np.zeros(len(parentinds[n]),dtype=int)
            for m in range(len(parentinds[n])):
                if len(parentinds[n][m])==1 and len(siblinginds[n][m])==1:
                    i[m]=ids[n-1][parentinds[n][m][0]]
                    ci[m]=cids[n-1][parentinds[n][m][0]]
                else:
                    i[m]=nextid
                    nextid=nextid+1
                    if len(parentinds[n][m])==0:
                        ci[m]=nextcid
                        nextcid=nextcid+1
                    else:
                        longestparentind=np.argmax(np.array(lengths[n-1])[parentinds[n][m]])
                        longestsiblingind=np.argmax(np.array(lengths[n])[siblinginds[n][m]])
                        if m==siblinginds[n][m][longestsiblingind]:
                            ci[m]=cids[n-1][parentinds[n][m][longestparentind]]
                        else:
                            ci[m]=nextcid
                            nextcid=nextcid+1

                    for pind in parentinds[n][m]:
                        edges=edges+[[ids[n-1][pind],i[m]]]
            ids=ids+[np.array(i)]
            cids=cids+[np.array(ci)]

        file=open(filebase+'lineids.dat','wb')
        pickle.dump((ids,cids,edges),file)
        file.close()

        stop=timeit.default_timer()
        print("line ancestry time: ", stop-start,flush=True)

        start=timeit.default_timer()
        pool=mp.Pool(threads,maxtasksperchild=1) 
        faces=pool.starmap(loadfaces, [(n,filebase,Ns,n0) for n in range(nt)])
        pool.close()
        stop=timeit.default_timer()
        print("faces load time: ", stop-start,flush=True)

        file=open(filebase+'faces.dat','wb')
        pickle.dump(faces,file)
        file.close()

        if rm:
            os.system('rm %sstates.dat'%filebase)


