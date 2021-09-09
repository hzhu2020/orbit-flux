import numpy as np
from time import time
from mpi4py import MPI
from parameters import xgc,bp_read,xgc_dir,orbit_dir,start_gstep,period,nsteps,nloops,\
                       sml_tri_psi_weighting,sml_grad_psitheta,Nr,Nz,gyro_E,\
                       diag_collision,diag_turbulence,diag_neutral,diag_source,diag_f0
import orbit
import grid
import main_loop

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
try:
  import cupy
  use_gpu=True
  if rank==0: print('Using CuPy for GPU acceleration...')
except:
  use_gpu=False

if nloops>nsteps:
  if rank==0: print('Warning: nloops>nsteps. Setting nloops=nsteps...')
  nloops=nsteps
#read from orbit.txt or orbit.bp
if rank==0: print('Reading orbit information...',flush=True)
if bp_read:
  orbit.readbp(orbit_dir,comm)
else:
  orbit.read(orbit_dir,comm)
#determine the range of nodes needed
if rank==0: print('Reading grid information...',flush=True)
grid.read(xgc,xgc_dir,Nr,Nz)
if diag_turbulence:
  if rank==0: print('Reading additional information for turbulence flux diagnostic...',flush=True)
  grid.additional_Bfield(xgc,xgc_dir,Nr,Nz)
  grid.grid_deriv_init(xgc_dir)
  grid.read_dpot_orb(orbit_dir)

if rank==0: print('Preparing orbit locations...',flush=True)
t_beg=time()
if use_gpu:
  grid.node_range_gpu(sml_tri_psi_weighting)
else:
  grid.node_range(sml_tri_psi_weighting)
t_end=time()
if rank==0: print('Preparing orbit locations took',(t_end-t_beg)/60.,'minutes',flush=True)
#set number of steps for each loop to avoid too large 
#memory usage when doing all time steps together
istep1,istep2=orbit.simple_partition(comm,nsteps,nloops)
#index loop starts here
for idx in range(1,6):
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
  #write a header to output
  if rank==0:
    output=open(orbit_dir+'/orbit_loss_'+source+'.txt','w')
    output.write('%8d%8d%8d%8d\n'%(orbit.nmu,orbit.nPphi,orbit.nH,nsteps))
    output.close()

  #main diagnosis starts here
  for iloop in range(nloops):
    nsteps_loop=istep2[iloop]-istep1[iloop]+1
    start_gstep_loop=start_gstep+istep1[iloop]*period
    t_beg=time()
    if(rank==0): print('Calculating',source+' flux, iloop=',iloop,flush=True)
    grid.readf0(xgc,xgc_dir,source,idx,start_gstep_loop,nsteps_loop,period)
    #prepare turbulence electric field
    if idx==1:
      if(rank==0): print('Reading turbulence electrostatic potential',flush=True)
      grid.read_dpot_turb(xgc,xgc_dir,start_gstep_loop,nsteps_loop,period)
      if (xgc=='xgc1')and(gyro_E):
        t1=time()
        itasks1,itasks2=orbit.simple_partition(comm,grid.nphi*nsteps_loop,size)
        grid.gyropot(use_gpu,nsteps_loop,itasks1[rank],itasks2[rank])
        grid.dpot_turb_rho=comm.allreduce(grid.dpot_turb_rho,op=MPI.SUM)
        t2=time()
        if(rank==0): print('Gyroaveraging potential took',(t2-t1)/60.,'min',flush=True)
      if(rank==0): print('Calculating electric fields',flush=True)
      if use_gpu:
        grid.Eturb_gpu(xgc,nsteps_loop,sml_grad_psitheta,False)
      else:
        grid.Eturb(xgc,nsteps_loop,sml_grad_psitheta,False)
    if use_gpu:
      dF_orb=np.zeros((grid.nphi,orbit.iorb2-orbit.iorb1+1,nsteps_loop),dtype=float)
      for iphi in range(grid.nphi):
        main_loop.copy_data(idx,nsteps_loop,iphi)
        dF_orb[iphi,:,:]=main_loop.dF_orb_main_gpu(orbit.iorb1,orbit.iorb2,nsteps_loop,idx)
        comm.barrier()
        main_loop.clear_data()
      dF_orb=np.mean(dF_orb,axis=0)
    else:
      dF_orb=np.zeros((orbit.iorb2-orbit.iorb1+1,nsteps_loop),dtype=float)
      for iorb in range(orbit.iorb1,orbit.iorb2+1):
        dF_orb[iorb-orbit.iorb1,:]=main_loop.dF_orb_main(iorb,nsteps_loop,idx)
    #write outputs
    iorb1_list=comm.gather(orbit.iorb1,root=0)
    iorb2_list=comm.gather(orbit.iorb2,root=0)
    if rank==0:
      norb=orbit.nmu*orbit.nPphi*orbit.nH
      dF_output=np.zeros((norb,),dtype=float)
      count=np.array(iorb2_list)-np.array(iorb1_list)+1
      displ=np.array(iorb1_list)-1
    else:
      dF_output,displ,count=[None]*3
    for istep in range(nsteps_loop):
      value=np.array(dF_orb[:,istep],order='C')
      comm.Gatherv(value,[dF_output,count,displ,MPI.DOUBLE],root=0)
      if rank==0:
        output=open(orbit_dir+'/orbit_loss_'+source+'.txt','a')
        count=0
        for i in range(norb):
          count=count+1
          output.write('%19.10E '%dF_output[i])
          if count%4==0: output.write('\n')
        if count%4!=0: output.write('\n')
        if((iloop==nloops-1)and(istep==nsteps_loop-1)): output.write('%8d'%-1)
        output.close()
    t_end=time()
    if rank==0: print(source,'flux, iloop=',iloop,'finished in',(t_end-t_beg)/60.,'minutes',flush=True)
  #end for iloop
comm.barrier()
#end for idx
