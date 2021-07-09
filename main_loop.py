def dF_orb_main(iorb,nsteps_loop,idx):
  import orbit,grid
  import numpy as np
  from math import floor
  from parameters import mi,qi,sml_dt
  p=np.zeros((3,),dtype=float)
  tmp=np.zeros((2,2,nsteps_loop),dtype=float)
  imu_orb=floor((iorb-1)/(orbit.nPphi*orbit.nH))+1
  mu=orbit.mu_orb[imu_orb-1]
  dF_orb=np.zeros((nsteps_loop,),dtype=float)
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
    itr=grid.itr_save[iorb-orbit.iorb1,it_orb]
    p[:]=grid.p_save[iorb-orbit.iorb1,it_orb,:]
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
            E[0,:]=grid.Er[node-1,:]
            E[1,:]=grid.Ez[node-1,:]
          else:
            E[0,:]=(grid.Er[node-1,:]*Bz+grid.Ez[node-1,:]*Br)/np.sqrt(Br**2+Bz**2)
            E[1,:]=(-grid.Er[node-1,:]*Br+grid.Ez[node-1,:]*Bz)/np.sqrt(Br**2+Bz**2)

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
          inode=node-grid.min_node
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
    dF_orb[:]=dF_orb[:]+df0g_orb[:]*(mi/np.pi/2)**1.5/1.6022E-19
  #end for it_orb
  return dF_orb
