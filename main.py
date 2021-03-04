import numpy as np
from math import floor
from mpi4py import MPI
from parameters import input_dir,ngroup,start_gstep,period,nsteps,sml_tri_psi_weighting,\
             Nr,Nz,qi,mi,sml_dt
import f0
import orbit
import grid

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

if (comm_size%ngroup!=0):
  if (comm_rank==0): print('Stop: number of processes must be divisible by ngroup!')
  exit()

group_size=int(comm_size/ngroup)
group_rank=int(comm_rank/group_size)
group_comm=MPI.Comm.Split(comm,group_rank)
if group_comm.Get_rank()==group_size-1:
  is_manager=True
else:
  is_manager=False

manager_ranks=(np.array(range(ngroup))+1)*group_size-1
manager_group=MPI.Group.Incl(MPI.Comm.Get_group(comm),manager_ranks)
manager_comm=MPI.Comm.Create_group(comm,manager_group)
worker_group=MPI.Group.Excl(MPI.Comm.Get_group(comm),manager_ranks)
worker_comm=MPI.Comm.Create_group(comm,worker_group)

#read from orbit.txt
orbit.read(comm,worker_comm,is_manager)
#determine the range of nodes needed
if not(is_manager):
  #workers read mesh information
  grid.read(input_dir,Nr,Nz)
  min_node=grid.nnode
  max_node=0
  norb=orbit.iorb2-orbit.iorb1+1
  for iorb in range(norb):
    for it_orb in range(orbit.steps_orb[iorb]):
      r=orbit.R_orb[iorb,it_orb]
      z=orbit.Z_orb[iorb,it_orb]
      itr,p=grid.search_tr2([r,z])
      if itr>0:
        for i in range(3):
          node=grid.nd[i,itr-1]
          if (node>max_node): max_node=node
          if (node<min_node): min_node=node
else:
  min_node=int(1e10)
  max_node=0

min_node=group_comm.allreduce(min_node,op=MPI.MIN)
max_node=group_comm.allreduce(max_node,op=MPI.MAX)

#main loop start here
for istep in range(nsteps):
  if (istep==0)and(comm_rank==0):
    output=open('orbit_loss_collision.txt','w')
    output.write('%8d%8d%8d%8d\n'%(orbit.nmu,orbit.nPphi,orbit.nH,nsteps))
    output.close()
  gstep=start_gstep+istep*period
  fname=input_dir+'/xgc.orbit.collision.'+'{:0>5d}'.format(gstep)+'.bp'
  #managers read f0
  if (is_manager):
    f0.read(fname,min_node,max_node)
    status=MPI.Status()

  dF_orb=np.zeros((orbit.iorb2-orbit.iorb1+1),dtype=float)
  if not(is_manager):
    for iorb in range(orbit.iorb1,orbit.iorb2+1):
      imu_orb=floor((iorb-1)/(orbit.nPphi*orbit.nH))+1
      mu=orbit.mu_orb[imu_orb-1]
      for it_orb in range(orbit.steps_orb[iorb-1]):
        df0g_orb=0.0
        r=orbit.R_orb[iorb-orbit.iorb1,it_orb]
        z=orbit.Z_orb[iorb-orbit.iorb1,it_orb]
        vp=orbit.vp_orb[iorb-orbit.iorb1,it_orb]
        itr,p=grid.search_tr2([r,z])
        if (sml_tri_psi_weighting)and(itr>0)and(max(p)<1.0): p=grid.t_coeff_mod([r,z],itr,p)
        if (itr>0):
          for i in range(3):
            node=grid.nd[i,itr-1]
            B=grid.b_interpol([grid.rz[node-1,0],grid.rz[node-1,1]])
            psi=grid.psi_interpol([grid.rz[node-1,0],grid.rz[node-1,1]])
            tempi=grid.tempi_interpol([grid.rz[node-1,0],grid.rz[node-1,1]])*1.6022E-19
            mu_n=2*mu*B/tempi
            vp_n=vp/np.sqrt(tempi/mi)
            if (mu_n<grid.f0_smu_max**2)and(vp_n<grid.f0_vp_max)and(vp_n>=-grid.f0_vp_max):
              wmu=np.zeros((2,),dtype=float)
              wvp=np.zeros((2,),dtype=float)
              smu=np.sqrt(mu_n)
              wmu[0]=smu/grid.f0_dsmu
              imu_f0=floor(wmu[0])
              wmu[1]=wmu[0]-float(imu_f0)
              wmu[0]=1.0-wmu[1]
              wvp[0]=vp_n/grid.f0_dvp
              ivp_f0=floor(wvp[0])
              wvp[1]=wvp[0]-float(ivp_f0)
              wvp[0]=1.0-wvp[1]
              #(different from Fortran version) let manager do the calculations
              data=np.array([node-min_node,imu_f0,ivp_f0+grid.f0_nvp,wmu[0],wmu[1],wvp[0],wvp[1]],dtype=float)
              group_comm.send(data,dest=group_size-1,tag=iorb)
              value=group_comm.recv(source=group_size-1,tag=iorb)
              df0g_orb=df0g_orb+value*p[i]/np.sqrt(2*mu*B)
        #end if itr
        df0g_orb=df0g_orb*orbit.dt_orb[iorb-1]/sml_dt
        dF_orb[iorb-orbit.iorb1]=dF_orb[iorb-orbit.iorb1]+df0g_orb*(mi/np.pi/2)**1.5/1.6022E-19
      #end for it_orb
    #end for iorb
    group_comm.send([-1,1,int],dest=group_size-1,tag=0)
  else:
    finished=np.zeros(group_size-1,dtype=int)
    while sum(finished)<group_size-1:
      data=group_comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
      tag=status.Get_tag()
      inq_id=status.Get_source()
      if tag>0:
        node=int(data[0])
        imu=int(data[1])
        ivp=int(data[2])
        wmu=np.zeros((2,),dtype=float)
        wvp=np.zeros((2,),dtype=float)
        wmu[0:2]=data[3:5]
        wvp[0:2]=data[5:7]
        tmp=np.zeros((2,2),dtype=float)
        tmp[0:2,0:2]=f0.df0g[ivp:ivp+2,node,imu:imu+2]
        value=tmp[0,0]*wvp[0]*wmu[0]+tmp[0,1]*wvp[0]*wmu[1]+tmp[1,0]*wvp[1]*wmu[0]+tmp[1,1]*wvp[1]*wmu[1]
        group_comm.send(value,dest=inq_id,tag=tag)
      if tag==0:
        data=int(data[0])
        if data!=-1: print('Something wrong with communications.')
        finished[inq_id]=1
  #write outputs
  iorb1_list=group_comm.gather(orbit.iorb1,root=group_size-1)
  iorb2_list=group_comm.gather(orbit.iorb2,root=group_size-1)
  if (is_manager):
    count=np.array(iorb2_list)-np.array(iorb1_list)+1
    count[group_size-1]=0
    displ=np.array(iorb1_list)-1
  else:
    displ,count=[None]*2
  group_comm.Gatherv(dF_orb,[dF_orb,count,displ,MPI.DOUBLE],root=group_size-1)
  if (is_manager):
    dF_orb=manager_comm.reduce(dF_orb,op=MPI.SUM)
    if(manager_comm.Get_rank()==0):
      output=open('orbit_loss_collision.txt','a')
      count=0
      for i in range(np.size(dF_orb)):
        count=count+1
        output.write('%19.10E '%dF_orb[i])
        if count%4==0: output.write('\n')
      if count%4!=0: output.write('\n')
      if istep==nsteps-1: output.write('%8d'%-1)
      output.close()
  comm.barrier()
#end for istep
