import numpy as np
from parameters import ngroup

def read(orbit_dir,comm):
  size=comm.Get_size()
  rank=comm.Get_rank()
  fname=orbit_dir+'/orbit.txt'
  global nmu,nH,nPphi,nt,iorb1,iorb2,\
        steps_orb,dt_orb,mu_orb,R_orb,Z_orb,vp_orb
  if rank==0:
    fid=open(fname,'r')
    nmu=int(fid.readline(8))
    nPphi=int(fid.readline(8))
    nH=int(fid.readline(8))
    nt=int(fid.readline(8))
    fid.readline(1)
    norb=nmu*nPphi*nH

    mu_orb=np.zeros((nmu,),dtype=float)
    dt_orb=np.zeros((norb,),dtype=float)
    steps_orb=np.zeros((norb,),dtype=int)

    for i in range(norb):
      value=fid.readline(8)
      steps_orb[i]=int(value)
      if ((i+1)%4)==0: fid.readline(1)
    if (norb%4)!=0: fid.readline(1)

    for i in range(norb):
      value=fid.readline(20)
      dt_orb[i]=float(value)
      if ((i+1)%4)==0: fid.readline(1)
    if (norb%4)!=0: fid.readline(1)
 
    for i in range(nmu):
      value=fid.readline(20)
      mu_orb[i]=float(value)
      if ((i+1)%4)==0: fid.readline(1)
    if (nmu%4)!=0: fid.readline(1)
  else:
    nmu,nPphi,nH,nt,mu_orb,dt_orb,steps_orb=[None]*7
  
  nmu,nPphi,nH,nt,mu_orb,dt_orb,steps_orb\
    =comm.bcast((nmu,nPphi,nH,nt,mu_orb,dt_orb,steps_orb),root=0)
  
  norb=nmu*nPphi*nH
  iorb1_list,iorb2_list=partition_orbit(norb,size)
  norb_list=iorb2_list-iorb1_list+1
  iorb1=iorb1_list[rank]
  iorb2=iorb2_list[rank]
  
  tmp=np.zeros((norb,nt),dtype=float,order='C')
  mynorb=iorb2-iorb1+1
  R_orb=np.zeros((mynorb,nt),dtype=float,order='C')
  Z_orb=np.zeros((mynorb,nt),dtype=float,order='C')
  vp_orb=np.zeros((mynorb,nt),dtype=float,order='C')

  if rank==0:
    count=0
    for iorb in range(norb):
      for it in range(nt):
        count=count+1
        value=fid.readline(20)
        tmp[iorb,it]=float(value)
        if count%4==0: fid.readline(1)
    if count%4!=0: fid.readline(1)
  comm.Scatterv([tmp,norb_list*nt],R_orb,root=0)
   
  if rank==0:
    count=0
    for iorb in range(norb):
      for it in range(nt):
        count=count+1
        value=fid.readline(20)
        tmp[iorb,it]=float(value)
        if count%4==0: fid.readline(1)
    if count%4!=0: fid.readline(1)
  comm.Scatterv([tmp,norb_list*nt],Z_orb,root=0)

  if rank==0:
    count=0
    for iorb in range(norb):
      for it in range(nt):
        count=count+1
        value=fid.readline(20)
        tmp[iorb,it]=float(value)
        if count%4==0: fid.readline(1)
    if count%4!=0: fid.readline(1)
  comm.Scatterv([tmp,norb_list*nt],vp_orb,root=0)

  if rank==0:
    value=fid.readline(8)
    if value!='      -1': print('Wrong end flag of orbit.txt')
    fid.close()

def readbp(orbit_dir,comm):
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
