import numpy as np
from parameters import ngroup

def read(orbit_dir,comm,worker_comm,is_manager):
  fname=orbit_dir+'/orbit.txt'
  global nmu,nH,nPphi,nt,iorb1,iorb2,\
        steps_orb,dt_orb,mu_orb,R_orb,Z_orb,vp_orb
  if comm.Get_rank()==0:
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
  iorb1_list,iorb2_list=partition_orbit(norb,comm.Get_size())
  norb_list=iorb2_list-iorb1_list+1

  if is_manager:
     iorb1=1
     iorb2=norb
  else:
     iorb1=iorb1_list[worker_comm.Get_rank()]
     iorb2=iorb2_list[worker_comm.Get_rank()]
  
  tmp=np.zeros((norb,nt),dtype=float,order='C')
  if not(is_manager):
    mynorb=iorb2-iorb1+1
    R_orb=np.zeros((mynorb,nt),dtype=float,order='C')
    Z_orb=np.zeros((mynorb,nt),dtype=float,order='C')
    vp_orb=np.zeros((mynorb,nt),dtype=float,order='C')

  if comm.Get_rank()==0:
    count=0
    for iorb in range(norb):
      for it in range(nt):
        count=count+1
        value=fid.readline(20)
        tmp[iorb,it]=float(value)
        if count%4==0: fid.readline(1)
    if count%4!=0: fid.readline(1)
  if not(is_manager): worker_comm.Scatterv([tmp,norb_list*nt],R_orb,root=0)
   
  if comm.Get_rank()==0:
    count=0
    for iorb in range(norb):
      for it in range(nt):
        count=count+1
        value=fid.readline(20)
        tmp[iorb,it]=float(value)
        if count%4==0: fid.readline(1)
    if count%4!=0: fid.readline(1)
  if not(is_manager): worker_comm.Scatterv([tmp,norb_list*nt],Z_orb,root=0)

  if comm.Get_rank()==0:
    count=0
    for iorb in range(norb):
      for it in range(nt):
        count=count+1
        value=fid.readline(20)
        tmp[iorb,it]=float(value)
        if count%4==0: fid.readline(1)
    if count%4!=0: fid.readline(1)
  if not(is_manager): worker_comm.Scatterv([tmp,norb_list*nt],vp_orb,root=0)

  if comm.Get_rank()==0:
    value=fid.readline(8)
    if value!='      -1': print('Wrong end flag of orbit.txt')
    fid.close()

def partition_orbit(norb,size):
    num_workers=size-ngroup
    sum_steps=sum(steps_orb)
    avg_steps=float(sum_steps)/float(num_workers)
    accuml_steps=0
    iorb1_list=np.zeros((num_workers,),dtype=int)
    iorb2_list=np.zeros((num_workers,),dtype=int)
    iorb1_list[:]=1
    iorb2_list[:]=norb
    iorb=0
    for ipe in range(num_workers-1):
      iorb=iorb+1
      accuml_steps=accuml_steps+steps_orb[iorb-1]
      while ((accuml_steps<avg_steps)and(norb-iorb>num_workers-ipe)):
        iorb=iorb+1
        accuml_steps=accuml_steps+steps_orb[iorb-1]
 
      accuml_steps=accuml_steps-avg_steps
      iorb2_list[ipe]=iorb
      iorb1_list[ipe+1]=iorb+1

    return iorb1_list,iorb2_list
