import numpy as np
from time import time
from math import floor
from mpi4py import MPI
from parameters import xgc_dir,orbit_dir,ngroup,start_gstep,period,nsteps,mpi_io_test,\
                       sml_tri_psi_weighting,sml_grad_psitheta,Nr,Nz,qi,mi,sml_dt,\
                       diag_collision,diag_turbulence,diag_neutral,diag_source
import f0
import orbit
import grid

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

if (comm_size==1):
  print('Stop: at least two processes are needed.')
  exit()
if (comm_size<=ngroup):
  if (comm_rank==0): print('Stop: number of processes must be larger than number of groups.')
  exit()
if (comm_size%ngroup!=0):
  if (comm_rank==0): print('Stop: number of processes must be divisible by ngroup!',flush=True)
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
orbit.read(orbit_dir,comm,worker_comm,is_manager)
#determine the range of nodes needed
if not(is_manager):
  #workers read mesh information
  grid.read(xgc_dir,Nr,Nz)
  if (diag_turbulence)and(worker_comm.Get_rank()==0): grid.grid_deriv_init(xgc_dir)
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

#index loop starts here
for idx in range(1,5):
  if (idx==1):
    source='turbulence'
    if not(diag_turbulence): continue
  if (idx==2):
    source='collision'
    if not(diag_collision): continue
  if (idx==3):
    source='source'
    if not(diag_source): continue
  if (idx==4):
    source='neutral'
    if not(diag_neutral): continue
  #write a header to output
  if comm_rank==0:
    output=open(orbit_dir+'/orbit_loss_'+source+'.txt','w')
    output.write('%8d%8d%8d%8d\n'%(orbit.nmu,orbit.nPphi,orbit.nH,nsteps))
    output.close()
    if (mpi_io_test):
      output=open(orbit_dir+'/mpi_io_'+source+'.txt','w')
      output.write('%8d%8d%8d%8d\n'%(orbit.nmu,orbit.nPphi,orbit.nH,nsteps))
      output.close()

  if not(is_manager)and(mpi_io_test):
    goutput=MPI.File.Open(worker_comm,orbit_dir+'/mpi_io_'+source+'.txt',MPI.MODE_WRONLY)
    offset_heading=33
    num_lines=floor(orbit.nmu*orbit.nPphi*orbit.nH/4)+1
    offset_step=orbit.nmu*orbit.nPphi*orbit.nH*20+num_lines

  #main diagnosis loop start here
  for istep in range(nsteps):
    gstep=start_gstep+istep*period
    if(comm_rank==0): print(source+' flux, gstep=',gstep,flush=True)
    if idx==1:
      fname=xgc_dir+'/xgc.f0.'+'{:0>5d}'.format(gstep)+'.bp'
    else:
      fname=xgc_dir+'/xgc.orbit.'+source+'.'+'{:0>5d}'.format(gstep)+'.bp'
    #managers read f0
    if(is_manager):
      t_beg=time()
      f0.read(idx,fname,min_node,max_node)
      status=MPI.Status()
    #workers prepare turbulence electric field
    elif idx==1:
      if worker_comm.Get_rank()==0:
        Er_node,Ez_node=grid.Eturb(xgc_dir,gstep,sml_grad_psitheta)
      else:
        Er_node,Ez_node=[None]*2
      Er_node,Ez_node=worker_comm.bcast((Er_node,Ez_node),root=0)

    dF_orb=np.zeros((orbit.iorb2-orbit.iorb1+1),dtype=float)
    if not(is_manager):
      if mpi_io_test: text=''
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
                #communication with manager
                data=np.array([node-min_node,imu_f0,ivp_f0+grid.f0_nvp],dtype=float)
                group_comm.send(data,dest=group_size-1,tag=iorb)
                tmp=group_comm.recv(source=group_size-1,tag=iorb)
                value=tmp[0,0]*wvp[0]*wmu[0]+tmp[0,1]*wvp[0]*wmu[1]+tmp[1,0]*wvp[1]*wmu[0]+tmp[1,1]*wvp[1]*wmu[1]
                df0g_orb=df0g_orb+value*p[i]/np.sqrt(2*mu*B)
          #end if itr
          df0g_orb=df0g_orb*orbit.dt_orb[iorb-1]/sml_dt
          dF_orb[iorb-orbit.iorb1]=dF_orb[iorb-orbit.iorb1]+df0g_orb*(mi/np.pi/2)**1.5/1.6022E-19
        #end for it_orb
        if (mpi_io_test):
          text=text+"{:19.10E}".format(dF_orb[iorb-orbit.iorb1])+' '
          if (iorb%4==0)or(iorb==orbit.nmu*orbit.nPphi*orbit.nH): text=text+'\n'
      #end for iorb
      if (mpi_io_test):
        data=np.empty(len(text),dtype=np.int8)
        data[:]=bytearray(text,encoding='utf-8')
        offset_local=(orbit.iorb1-1)*20+floor((orbit.iorb1-1)/4)
        goutput.Write_at_all(offset_heading+offset_step*istep+offset_local,data)
      #notify manager that work is done
      group_comm.send([-1,1,int],dest=group_size-1,tag=0)
    else:#if manager
      finished=np.zeros(group_size-1,dtype=int)
      while sum(finished)<group_size-1:
        data=group_comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        tag=status.Get_tag()
        inq_id=status.Get_source()
        if tag>0:
          node=int(data[0])
          imu=int(data[1])
          ivp=int(data[2])
          tmp=np.zeros((2,2),dtype=float)
          tmp[0:2,0:2]=f0.df0g[ivp:ivp+2,node,imu:imu+2]
          group_comm.send(tmp,dest=inq_id,tag=tag)
        if tag==0:
          node=int(data[0])
          if node!=-1: print('Something wrong with communications.',flush=True)
          finished[inq_id]=1
      t_end=time()
      print('group',manager_comm.Get_rank(),'finished in cpu time',(t_end-t_beg)/60.0,'min',flush=True)
    #end if not manager
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
        output=open(orbit_dir+'/orbit_loss_'+source+'.txt','a')
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
#end for idx
if not(is_manager)and(mpi_io_test): goutput.Close()
