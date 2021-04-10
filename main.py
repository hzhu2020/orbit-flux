import numpy as np
from time import time
from math import floor
from mpi4py import MPI
from parameters import bp_read,xgc_dir,orbit_dir,start_gstep,period,nsteps,nloops,\
                       sml_tri_psi_weighting,sml_grad_psitheta,Nr,Nz,qi,mi,sml_dt,\
                       diag_collision,diag_turbulence,diag_neutral,diag_source
import orbit
import grid

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
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
grid.read(xgc_dir,Nr,Nz)
if diag_turbulence:
  if rank==0: print('Reading additional information for turbulence diagnosis...',flush=True)
  grid.additional_Bfield(xgc_dir,Nr,Nz)
  if rank==0:
    grid.grid_deriv_init(xgc_dir)
    grid.read_dpot_orb(orbit_dir)

if rank==0: print('Preparing orbit locations...',flush=True)
t_beg=time()
min_node,max_node,itr_save,p_save=grid.node_range(sml_tri_psi_weighting)
t_end=time()
if rank==0: print('Preparing orbit locations took',(t_end-t_beg)/60.,'minutes',flush=True)
#set number of steps for each loop to avoid too large 
#memory usage when doing all time steps together
if rank==0:
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
  if rank==0:
    output=open(orbit_dir+'/orbit_loss_'+source+'.txt','w')
    output.write('%8d%8d%8d%8d\n'%(orbit.nmu,orbit.nPphi,orbit.nH,nsteps))
    output.close()

  #main diagnosis starts here
  for iloop in range(nloops):
    nsteps_loop=istep2[iloop]-istep1[iloop]+1
    start_gstep_loop=start_gstep+istep1[iloop]*period
    tmp=np.zeros((2,2,nsteps_loop),dtype=float)
    p=np.zeros((3,),dtype=float)
    t_beg=time()
    if(rank==0): print('Calculating',source+' flux, iloop=',iloop,flush=True)
    grid.readf0(xgc_dir,source,idx,start_gstep_loop,nsteps_loop,period,min_node,max_node)
    #prepare turbulence electric field
    if idx==1:
      if rank==0:
        Er_node,Ez_node=grid.Eturb(xgc_dir,start_gstep_loop,nsteps_loop,period,sml_grad_psitheta,False)
      else:
        Er_node,Ez_node=[None]*2
      Er_node,Ez_node=comm.bcast((Er_node,Ez_node),root=0)

    dF_orb=np.zeros((orbit.iorb2-orbit.iorb1+1,nsteps_loop),dtype=float)
    for iorb in range(orbit.iorb1,orbit.iorb2+1):
      imu_orb=floor((iorb-1)/(orbit.nPphi*orbit.nH))+1
      mu=orbit.mu_orb[imu_orb-1]
      for it_orb in range(orbit.steps_orb[iorb-1]):
        df0g_orb=np.zeros((nsteps_loop,),dtype=float)
        if idx==1:
          E=np.zeros((2,nsteps_loop),dtype=float)
          dxdt=np.zeros((2,nsteps_loop),dtype=float)
          dvpdt=np.zeros((nsteps_loop,),dtype=float)
          F_node=np.zeros((3,nsteps_loop),dtype=float)
          dFdvp=np.zeros((nsteps_loop,),dtype=float)

        r=orbit.R_orb[iorb-orbit.iorb1,it_orb]
        z=orbit.Z_orb[iorb-orbit.iorb1,it_orb]
        vp=orbit.vp_orb[iorb-orbit.iorb1,it_orb]
        itr=itr_save[iorb-orbit.iorb1,it_orb]
        p[:]=p_save[iorb-orbit.iorb1,it_orb,:]
        if (itr>0):
          for i in range(3):
            node=grid.nd[i,itr-1]
            Br=grid.B[node-1,0]
            Bz=grid.B[node-1,1]
            Bphi=grid.B[node-1,2]
            B=np.sqrt(Br**2+Bz**2+Bphi**2)
            tempi=grid.tempi[node-1]*1.6022E-19
            mu_n=2*mu*B/tempi
            vp_n=vp/np.sqrt(tempi/mi)
            if idx==1:
              rho=mi*vp/qi/B
              D=1.0/(1.0+rho*grid.nb_curl_nb[node-1])
              curlbr=grid.curlbr[node-1]
              curlbz=grid.curlbz[node-1]
              if grid.basis[node-1]==1:
                E[0,:]=Er_node[node-1,:]
                E[1,:]=Ez_node[node-1,:]
              else:
                E[0,:]=(Er_node[node-1,:]*Bz+Ez_node[node-1,:]*Br)/np.sqrt(Br**2+Bz**2)
                E[1,:]=(-Er_node[node-1,:]*Br+Ez_node[node-1,:]*Bz)/np.sqrt(Br**2+Bz**2)

              dxdt[0,:]=dxdt[0,:]+p[i]*D*(-Bphi)*E[1,:]/B**2
              dxdt[1,:]=dxdt[1,:]+p[i]*D*(+Bphi)*E[0,:]/B**2
              dvpdt[:]=dvpdt[:]+p[i]*D*qi/mi*\
                   (E[0,:]*(Br/B+rho*grid.curlbr[node-1])+E[1,:]*(Bz/B+rho*grid.curlbz[node-1]))
            #end if idx
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
              imu=imu_f0
              ivp=ivp_f0+grid.f0_nvp
              inode=node-min_node
              tmp[0:2,0:2,:]=grid.df0g[ivp:ivp+2,inode,imu:imu+2,:]
              value=tmp[0,0,:]*wvp[0]*wmu[0]+tmp[0,1,:]*wvp[0]*wmu[1]\
                   +tmp[1,0,:]*wvp[1]*wmu[0]+tmp[1,1,:]*wvp[1]*wmu[1]
              if idx==1:
                F_node[i,:]=value[:]/np.sqrt(2*mu*B)
                dFdvp[:]=dFdvp[:]+(wmu[0]*(tmp[1,0,:]-tmp[0,0,:])+wmu[1]*(tmp[1,1,:]-tmp[0,1,:]))\
                      *p[i]/np.sqrt(2*mu*B)/(grid.f0_dvp*np.sqrt(tempi/mi))
              else:
                df0g_orb[:]=df0g_orb[:]+value[:]*p[i]/np.sqrt(2*mu*B)
          #end for i
          if idx==1:
            grad_F=grid.gradF_orb(F_node,itr,nsteps_loop)
            df0g_orb[:]=-dxdt[0,:]*grad_F[0,:]-dxdt[1,:]*grad_F[1,:]-dvpdt[:]*dFdvp[:]
            df0g_orb[:]=df0g_orb[:]*sml_dt
        #end if itr
        df0g_orb[:]=df0g_orb[:]*orbit.dt_orb[iorb-1]/sml_dt
        dF_orb[iorb-orbit.iorb1,:]=dF_orb[iorb-orbit.iorb1,:]+df0g_orb[:]*(mi/np.pi/2)**1.5/1.6022E-19
      #end for it_orb
    #end for iorb
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
    if rank==0: print(source,'diagnosis, iloop=',iloop,'finished in',(t_end-t_beg)/60.,'minutes',flush=True)
  #end for iloop
comm.barrier()
#end for idx
