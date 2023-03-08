import numpy as np
from time import time
from mpi4py import MPI
from parameters import xgc,xgc_dir,orbit_dir,start_gstep,period,nsteps,nloops,split_nphi,\
                       sml_tri_psi_weighting,sml_grad_psitheta,Nr,Nz,gyro_E,use_ff,\
                       diag_collision,diag_turbulence,diag_neutral,diag_source,diag_f0,diag_df0
import adios2
import orbit
import grid
import main_loop

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#test GPU
try:
  import cupy
  use_gpu=True
  if rank==0: print('Using CuPy for GPU acceleration...')
except:
  use_gpu=False

if nloops>nsteps:
  if rank==0: print('Warning: nloops>nsteps. Setting nloops=nsteps...')
  nloops=nsteps

if (xgc=='xgc1')and(diag_turbulence)and(split_nphi>1)and(np.mod(size,split_nphi)==0):
  if rank==0: print('Split communicator into',split_nphi,'plane communicators to accelerate turb calculation.',flush=True)
  turb_split_phi=True
  plane_id=rank%split_nphi
  plane_rank=int(float(rank)/float(split_nphi))
  plane_comm=comm.Split(color=plane_id,key=plane_rank)
  intpl_comm=comm.Split(color=plane_rank,key=plane_id)
else:
  turb_split_phi=False
  plane_id=0
  plane_rank=rank
  plane_comm=comm
  intpl_comm=MPI.COMM_SELF

if rank==0: print('Reading orbit information...',flush=True)
orbit.read(orbit_dir,plane_comm)
#test adios2 mpi
try:
  testmpi=adios2.open(orbit_dir+'/orbit.bp','r',comm)
  adios2_mpi=True
  testmpi.close()
except:
  if (rank==0): print('Adios2 does not support MPI.',flush=True)
  adios2_mpi=False

#determine the range of nodes needed
if rank==0: print('Reading grid information...',flush=True)
grid.read(xgc,use_ff,xgc_dir,Nr,Nz)
if diag_turbulence:
  if rank==0: print('Reading additional information for turbulence flux diagnostic...',flush=True)
  itask1,itask2=orbit.simple_partition(comm,3,size)
  grid.additional_Bfield(xgc,xgc_dir,Nr,Nz,itask1[rank],itask2[rank],comm,MPI.SUM)
  grid.grid_deriv_init(xgc_dir)
  grid.read_dpot_orb(orbit_dir,rank)

if rank==0: print('Preparing orbit locations...',flush=True)
t_beg=time()
if use_gpu:
  grid.node_range_gpu(sml_tri_psi_weighting)
else:
  grid.node_range(sml_tri_psi_weighting)
#also determine the range of nodes needed for spatial derivative
if diag_turbulence:
  if (xgc=='xgc1')and(use_ff): grid.interpE_node_range()
  grid.deriv_node_range(xgc,use_ff)
  grid.deriv_min_node=comm.allreduce(grid.deriv_min_node,op=MPI.MIN)
  grid.deriv_max_node=comm.allreduce(grid.deriv_max_node,op=MPI.MAX)
t_end=time()
if rank==0: print('Preparing orbit locations took',(t_end-t_beg)/60.,'minutes',flush=True)
#set number of steps for each loop to avoid too large 
#memory usage when doing all time steps together
istep1,istep2=orbit.simple_partition(comm,nsteps,nloops)
#index loop starts here
for idx in range(1,7):
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
  if (idx==5):
    source='f0'
    if not(diag_f0): continue
  if (idx==6):
    source='df0'
    if not(diag_df0): continue
  if (turb_split_phi)and(idx>1):
    if rank==0: print('Use single plane communicator for non xgc1 turb calculation.',flush=True)
    turb_split_phi=False
    plane_id=0
    plane_comm.Free()
    intpl_comm.Free()
    plane_rank=rank
    plane_comm=comm
    intpl_comm=MPI.COMM_SELF
    if rank==0: print('Reading orbit information...',flush=True)
    orbit.read(orbit_dir,plane_comm)
    if rank==0: print('Preparing orbit locations...',flush=True)
    if use_gpu:
      grid.node_range_gpu(sml_tri_psi_weighting)
    else:
      grid.node_range(sml_tri_psi_weighting)

  adios2_write=True
  if (plane_id==0)and(adios2_mpi):
    output=adios2.open(orbit_dir+'/orbit_loss_'+source+'.bp','w',plane_comm)
  elif (rank==0)and(not adios2_mpi):
    output=adios2.open(orbit_dir+'/orbit_loss_'+source+'.bp','w')
  else:
    adios2_write=False
  #write a header to output
  if adios2_write:
    output.write('nmu',np.array(orbit.nmu))
    output.write('nPphi',np.array(orbit.nPphi))
    output.write('nH',np.array(orbit.nH))
    output.write('nsteps',np.array(nsteps))
  #main diagnosis starts here
  for iloop in range(nloops):
    nsteps_loop=istep2[iloop]-istep1[iloop]+1
    start_gstep_loop=start_gstep+istep1[iloop]*period
    t_beg=time()
    if(rank==0): print('Calculating',source+' flux, iloop=',iloop,flush=True)
    grid.readf0(xgc,xgc_dir,idx,start_gstep_loop,nsteps_loop,period)
    #prepare turbulence electric field
    if idx==1:
      if(rank==0): print('Reading turbulence electrostatic potential',flush=True)
      grid.read_dpot_turb(xgc,xgc_dir,start_gstep_loop,nsteps_loop,period)
      if (xgc=='xgc1')and(gyro_E):
        t1=time()
        from parameters import nrho
        itasks1,itasks2=orbit.simple_partition(comm,grid.nphi*nsteps_loop*nrho,size)
        grid.gyropot(use_gpu,nsteps_loop,itasks1[rank],itasks2[rank])
        grid.dpot_turb_rho=comm.allreduce(grid.dpot_turb_rho,op=MPI.SUM)
        t2=time()
        if(rank==0): print('Gyroaveraging potential took',(t2-t1)/60.,'min',flush=True)
      if use_gpu:
        grid.Eturb_gpu(xgc,use_ff,gyro_E,nsteps_loop,sml_grad_psitheta,False)
      else:
        grid.Eturb(xgc,use_ff,gyro_E,nsteps_loop,sml_grad_psitheta,False)
      if(rank==0): print('Finished calculating electric fields',flush=True)
    if idx==6:
      dF_orb=np.zeros((orbit.iorb2-orbit.iorb1+1,nsteps_loop),dtype=float)
      for iorb in range(orbit.iorb1,orbit.iorb2+1):
        dF_orb[iorb-orbit.iorb1,:]=main_loop.dF_in_out(iorb,nsteps_loop)
    else:
      dF_orb=np.zeros((grid.nphi,orbit.iorb2-orbit.iorb1+1,nsteps_loop),dtype=float)
      if use_gpu:
        if (xgc=='xgc1')and(idx==1):
          if (turb_split_phi):
            iphi1,iphi2=orbit.simple_partition(comm,grid.nphi,split_nphi)
            iphi1=iphi1[plane_id]
            iphi2=iphi2[plane_id]
          else:
            iphi1=0
            iphi2=grid.nphi-1
        else:
          iphi1=0
          iphi2=0
        for iphi in range(iphi1,iphi2+1):
          if (xgc=='xgc1')and(idx==1):
            grid.readf0_xgc1_turb(iphi,iphi1,xgc_dir,start_gstep_loop,nsteps_loop,period,use_ff)
          main_loop.copy_data(idx,nsteps_loop,iphi)
          dF_orb[iphi,:,:]=main_loop.dF_orb_main_gpu(orbit.iorb1,orbit.iorb2,nsteps_loop,idx)
          plane_comm.barrier()
          main_loop.clear_data()
      else:
        #for XGC1 turbulence diagnosis, may need to limit memory usage
        if (xgc=='xgc1')and(idx==1):
          from parameters import nsteps_phi
        else:
          nsteps_phi=1
        iphi1,iphi2=orbit.simple_partition(comm,grid.nphi,nsteps_phi)
        for istep in range(nsteps_phi):
          myiphi1=iphi1[istep]
          myiphi2=iphi2[istep]
          if (xgc=='xgc1')and(idx==1):
            grid.readf0_xgc1_turb2(myiphi1,myiphi2,xgc_dir,start_gstep_loop,nsteps_loop,period,use_ff)
          for iorb in range(orbit.iorb1,orbit.iorb2+1):
            dF_orb[myiphi1:myiphi2+1,iorb-orbit.iorb1,:]=main_loop.dF_orb_main(iorb,nsteps_loop,idx,myiphi1,myiphi2)
      #end if use_gpu
      dF_orb=np.mean(dF_orb,axis=0)
      dF_orb=intpl_comm.allreduce(dF_orb,op=MPI.SUM)
    #write outputs
    iorb1_list=plane_comm.gather(orbit.iorb1,root=0)
    iorb2_list=plane_comm.gather(orbit.iorb2,root=0)
    norb=orbit.nmu*orbit.nPphi*orbit.nH
    for istep in range(nsteps_loop):
      if plane_id!=0: continue
      value=np.array(dF_orb[:,istep],order='C')
      if (rank==0)and(not adios2_mpi):
        dF_output=np.zeros((norb,),dtype=float)
        count=np.array(iorb2_list)-np.array(iorb1_list)+1
        displ=np.array(iorb1_list)-1
      else:
        dF_output,displ,count=[None]*3
      if (not adios2_mpi): plane_comm.Gatherv(value,[dF_output,count,displ,MPI.DOUBLE],root=0)
      if (adios2_mpi):
        shape=np.array([norb,],dtype=int)
        start=np.array([orbit.iorb1-1,],dtype=int)
        count=np.array([orbit.iorb2-orbit.iorb1+1,],dtype=int)
        output.write('dF_orb',value,shape,start,count,end_step=True)
      elif (rank==0):
        shape=np.array([norb,],dtype=int)
        start=np.array([0,],dtype=int)
        count=np.array([norb,],dtype=int)
        output.write('dF_orb',dF_output,shape,start,count,end_step=True)
    t_end=time()
    if rank==0: print(source,'flux, iloop=',iloop,'finished in',(t_end-t_beg)/60.,'minutes',flush=True)
  #end for iloop
  if adios2_write: output.close()
comm.barrier()
#end for idx
