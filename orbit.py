import numpy as np

def read(orbit_dir,comm):
  import adios2
  rank=comm.Get_rank()
  size=comm.Get_size()
  fname=orbit_dir+'/orbit.bp'
  global nmu,nH,nPphi,nt,iorb1,iorb2,\
        steps_orb,dt_orb,mu_orb,R_orb,Z_orb,vp_orb
  if rank==0:
    fid=adios2.open(fname,'r')
    nmu=fid.read('nmu')
    nPphi=fid.read('nPphi')
    nH=fid.read('nH')
    nt=fid.read('nt')
    norb=nmu*nPphi*nH
    mu_orb=np.zeros((nmu,),dtype=float)
    dt_orb=np.zeros((norb,),dtype=float)
    steps_orb=np.zeros((norb,),dtype=int)
    mu_orb[:]=fid.read('mu_orb')
    dt_orb[:]=fid.read('dt_orb')
    steps_orb[:]=fid.read('steps_orb')
    fid.close() 
  else:
    nmu,nPphi,nH,nt,mu_orb,dt_orb,steps_orb=[None]*7
  
  nmu,nPphi,nH,nt,mu_orb,dt_orb,steps_orb\
    =comm.bcast((nmu,nPphi,nH,nt,mu_orb,dt_orb,steps_orb),root=0)
  
  norb=nmu*nPphi*nH
  iorb1_list,iorb2_list=partition_orbit(norb,size)
  norb_list=iorb2_list-iorb1_list+1
  iorb1=iorb1_list[rank]
  iorb2=iorb2_list[rank]
  mynorb=iorb2-iorb1+1
  R_orb=np.zeros((mynorb,nt),dtype=float,order='C')
  Z_orb=np.zeros((mynorb,nt),dtype=float,order='C')
  vp_orb=np.zeros((mynorb,nt),dtype=float,order='C')
  fid=adios2.open(fname,'r')
  R_orb[:,:]=fid.read('R_orb',start=[iorb1-1,0],count=[mynorb,nt]) 
  Z_orb[:,:]=fid.read('Z_orb',start=[iorb1-1,0],count=[mynorb,nt]) 
  vp_orb[:,:]=fid.read('vp_orb',start=[iorb1-1,0],count=[mynorb,nt]) 
  fid.close()

def partition_orbit(norb,size):
    sum_steps=sum(steps_orb)
    avg_steps=float(sum_steps)/float(size)
    accuml_steps=0
    iorb1_list=np.zeros((size,),dtype=int)
    iorb2_list=np.zeros((size,),dtype=int)
    iorb1_list[:]=1
    iorb2_list[:]=norb
    iorb=0
    for ipe in range(size-1):
      iorb=iorb+1
      accuml_steps=accuml_steps+steps_orb[iorb-1]
      while ((accuml_steps<avg_steps)and(norb-iorb>size-ipe)):
        iorb=iorb+1
        accuml_steps=accuml_steps+steps_orb[iorb-1]
 
      accuml_steps=accuml_steps-avg_steps
      iorb2_list[ipe]=iorb
      iorb1_list[ipe+1]=iorb+1

    return iorb1_list,iorb2_list

def simple_partition(comm,nsteps,nloops):
  if comm.Get_rank()==0:
    nsteps_avg=int(nsteps/nloops)
    nsteps_last=nsteps-nsteps_avg*(nloops-1)
    nsteps_list=np.zeros((nloops,),dtype=int)
    nsteps_list[:]=nsteps_avg
    if nsteps_last>nsteps_avg:
      for iloop in range(nsteps_last-nsteps_avg): nsteps_list[iloop]=nsteps_list[iloop]+1
    istep1=np.zeros((nloops,),dtype=int)
    istep2=np.zeros((nloops,),dtype=int)
    for iloop in range(nloops): istep1[iloop]=int(sum(nsteps_list[0:iloop]))
    istep2=istep1+nsteps_list-1

  else:
    istep1,istep2=[None]*2
  istep1=comm.bcast(istep1,root=0)
  istep2=comm.bcast(istep2,root=0)
  return istep1,istep2
