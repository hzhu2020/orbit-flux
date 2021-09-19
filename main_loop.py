def dF_orb_main(iorb,nsteps_loop,idx):
  import orbit,grid
  import numpy as np
  from math import floor
  from parameters import mi,qi,sml_dt,xgc,gyro_E,use_ff
  if (idx==1)and(xgc=='xgc1')and(gyro_E):
    from parameters import nrho
  else:
    nrho=0
  if nrho>0:
    from parameters import rhomax
    drho=rhomax/float(nrho)
  p=np.zeros((3,),dtype=float)
  tmp=np.zeros((2,2,grid.nphi,nsteps_loop),dtype=float)
  imu_orb=floor((iorb-1)/(orbit.nPphi*orbit.nH))+1
  mu=orbit.mu_orb[imu_orb-1]
  dF_orb=np.zeros((grid.nphi,nsteps_loop),dtype=float)
  for it_orb in range(orbit.steps_orb[iorb-1]):
    df0g_orb=np.zeros((grid.nphi,nsteps_loop),dtype=float)
    if idx==1:
      wrho=np.zeros((2,),dtype=float)
      E=np.zeros((grid.nphi,3,nsteps_loop),dtype=float)
      dxdt=np.zeros((grid.nphi,3,nsteps_loop),dtype=float)
      dvpdt=np.zeros((grid.nphi,nsteps_loop),dtype=float)
      F_node=np.zeros((grid.nphi,3,nsteps_loop),dtype=float)
      dFdvp=np.zeros((grid.nphi,nsteps_loop),dtype=float)
      dFdRphi=np.zeros((grid.nphi,nsteps_loop),dtype=float)
      if (xgc=='xgc1')and(use_ff):
        Br_l=0.
        Bz_l=0.
        Bphi_l=0.
        gradParF=np.zeros((grid.nphi,nsteps_loop),dtype=float)

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
          if nrho>0:
            rho_perp=min(rhomax,np.sqrt(2*mi*mu/B)/qi)
            rhon=rho_perp/(drho+1E-10)#avoid division by zero
            irho0=min(floor(rhon),nrho-1)
            irho1=irho0+1
            wrho[1]=rhon-float(irho0)
            wrho[0]=1.0-wrho[1]
          else:
            irho0=0
            irho1=0
            wrho[0]=1.0
            wrho[1]=0.0
          E[:,2,:]=wrho[0]*grid.Ephi[:,node-1,:,irho0]+wrho[1]*grid.Ephi[:,node-1,:,irho1]
          if grid.basis[node-1]==1:
            E[:,0,:]=wrho[0]*grid.Er[:,node-1,:,irho0]+wrho[1]*grid.Er[:,node-1,:,irho1]
            E[:,1,:]=wrho[0]*grid.Ez[:,node-1,:,irho0]+wrho[1]*grid.Ez[:,node-1,:,irho1]
          else:
            E[:,0,:]=wrho[0]*(grid.Er[:,node-1,:,irho0]*Bz+grid.Ez[:,node-1,:,irho0]*Br)/np.sqrt(Br**2+Bz**2)\
                    +wrho[1]*(grid.Er[:,node-1,:,irho1]*Bz+grid.Ez[:,node-1,:,irho1]*Br)/np.sqrt(Br**2+Bz**2)
            E[:,1,:]=wrho[0]*(-grid.Er[:,node-1,:,irho0]*Br+grid.Ez[:,node-1,:,irho0]*Bz)/np.sqrt(Br**2+Bz**2)\
                    +wrho[1]*(-grid.Er[:,node-1,:,irho1]*Br+grid.Ez[:,node-1,:,irho1]*Bz)/np.sqrt(Br**2+Bz**2)

          dxdt[:,0,:]=dxdt[:,0,:]+p[i]*D*(Bz*E[:,2,:]-Bphi*E[:,1,:])/B**2
          dxdt[:,1,:]=dxdt[:,1,:]+p[i]*D*(Bphi*E[:,0,:]-Br*E[:,2,:])/B**2
          dxdt[:,2,:]=dxdt[:,2,:]+p[i]*D*(Br*E[:,1,:]-Bz*E[:,0,:])/B**2
          dvpdt[:,:]=dvpdt[:,:]+p[i]*D*qi/mi*\
               (E[:,0,:]*(Br/B+rho*grid.curlbr[node-1])\
               +E[:,1,:]*(Bz/B+rho*grid.curlbz[node-1])\
               +E[:,2,:]*(Bphi/B+rho*grid.curlbphi[node-1]))
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
          tmp[0:2,0:2,:,:]=grid.df0g[ivp:ivp+2,inode,imu:imu+2,:,:]
          value=tmp[0,0,:,:]*wvp[0]*wmu[0]+tmp[0,1,:,:]*wvp[0]*wmu[1]\
               +tmp[1,0,:,:]*wvp[1]*wmu[0]+tmp[1,1,:,:]*wvp[1]*wmu[1]
          if idx==1:
            F_node[:,i,:]=value[:,:]/np.sqrt(2*mu*B)
            dFdvp[:,:]=dFdvp[:,:]+(wmu[0]*(tmp[1,0,:,:]-tmp[0,0,:,:])+wmu[1]*(tmp[1,1,:,:]-tmp[0,1,:,:]))\
                   *p[i]/np.sqrt(2*mu*B)/(grid.f0_dvp*np.sqrt(tempi/mi))
            if (xgc=='xgc1')and(use_ff):
              Br_l=Br_l+p[i]*Br
              Bz_l=Bz_l+p[i]*Bz
              Bphi_l=Bphi_l+p[i]*Bphi
              gradParF[:,:]=gradParF[:,:]+p[i]*grid.gradParF_ff(node,imu,ivp,nsteps_loop,wmu,wvp)/np.sqrt(2*mu*B)
            if (xgc=='xgc1')and(not use_ff):
              dphi=2*np.pi/float(grid.nphi*grid.nwedge)
              for iphi in range(grid.nphi):
                iphip1=(iphi+1)%grid.nphi
                iphim1=(iphi-1)%grid.nphi
                dFdRphi[iphi,:]=dFdRphi[iphi,:]+p[i]*\
                                (F_node[iphip1,i,:]-F_node[iphim1,i,:])/2/dphi/grid.rz[node-1,0]
          else:
            df0g_orb[:,:]=df0g_orb[:,:]+value[:,:]*p[i]/np.sqrt(2*mu*B)
      #end for i
      if idx==1:
        grad_F=grid.gradF_orb(F_node,itr,nsteps_loop)
        if (xgc=='xgc1')and(use_ff):
          B_l=np.sqrt(Br_l**2+Bz_l**2+Bphi_l**2)
          if B_l>1E-4: dFdRphi[:,:]=(gradParF[:,:]*B_l-grad_F[:,0,:]*Br_l-grad_F[:,1,:]*Bz_l)/Bphi_l
        df0g_orb[:,:]=-dxdt[:,0,:]*grad_F[:,0,:]-dxdt[:,1,:]*grad_F[:,1,:]\
                      -dxdt[:,2,:]*dFdRphi[:,:]-dvpdt[:,:]*dFdvp[:,:]
        df0g_orb[:,:]=df0g_orb[:,:]*sml_dt
    #end if itr
    df0g_orb[:,:]=df0g_orb[:,:]*orbit.dt_orb[iorb-1]/sml_dt
    dF_orb[:,:]=dF_orb[:,:]+df0g_orb[:,:]*(mi/np.pi/2)**1.5/1.6022E-19
  #end for it_orb
  return np.mean(dF_orb,axis=0)

def dF_orb_main_gpu(iorb1,iorb2,nsteps_loop,idx):
  import orbit,grid
  import numpy as np
  import cupy as cp
  from math import floor
  from parameters import mi,qi,sml_dt,xgc,gyro_E,use_ff
  dF_orb_kernel=cp.RawKernel(r'''
  extern "C" __device__
  void gradF_orb(double* F,int itr,int num_tri,int* nd,double* rz,double* dFdx)
  {
    double r[3],z[3],xsj;
    int node;
    for(int k=0;k<3;k++){
      node=nd[k*num_tri+itr-1];
      r[k]=rz[(node-1)*2+0];
      z[k]=rz[(node-1)*2+1];
    }
    xsj=r[0]*(z[1]-z[2])+r[1]*(z[2]-z[0])+r[2]*(z[0]-z[1]);
    dFdx[0]=F[0]*(z[1]-z[2])+F[1]*(z[2]-z[0])+F[2]*(z[0]-z[1]);
    dFdx[0]=dFdx[0]/xsj;
    dFdx[1]=F[0]*(r[2]-r[1])+F[1]*(r[0]-r[2])+F[2]*(r[1]-r[0]);
    dFdx[1]=dFdx[1]/xsj;
  }
  extern "C" __global__
  void dF_orb_main(double* dF_orb,double* mu_orb,int nsteps_loop,int* steps_orb,int nPphi,int nH,\
    int nt,int nblocks_max,int mynorb,int idx,double* rt,double* zt,double* vpt,int* itr_save,\
    double* p_save,int* nd,int num_tri,double* B,double* Ti,double mi,double qi,double* nb_curl_nb,\
    double* curlbr,double* curlbz,double* curlbphi,int* basis,double* Er,double* Ez,double* Ephi,
    double f0_smu_max,double f0_vp_max,double f0_dsmu,double f0_dvp,int f0_nvp,int f0_nmu,int min_node,\
    int max_node,double* df0g,double* dFdphi,double sml_dt,double* dt_orb,double* rz,int iorb1,\
    int nphi,int nrho,double rhomax,double* dFdpara,bool use_ff)
  {
    int it_orb,istep,iorb,imu_orb,itr,node,imu_f0,ivp_f0,imu,ivp,inode,nnode,nvp,nmu;
    double p[3],tmp[2][2],r,z,vp,Br,Bz,Bphi,Bmag,tempi,mu_n,vp_n,rho,D,wmu[2],wvp[2],smu;
    double mu,df0g_orb,dvpdt,dFdvp,dFdRphi,value,E[3],dxdt[3],F_node[3],grad_F[2];
    double drho,rho_perp,rhon,wrho[2];
    double gradParF,Br_l,Bz_l,Bphi_l,Bmag_l;
    int irho0,irho1;
    iorb=blockIdx.x/nsteps_loop;
    istep=blockIdx.x-iorb*nsteps_loop;
    it_orb=threadIdx.x;
    while(iorb<mynorb)
    {
    if (it_orb>=steps_orb[iorb]){
      iorb=iorb+nblocks_max;
      continue;
    }
    if (nrho>0) drho=rhomax/double(nrho);
    imu_orb=(iorb+iorb1-1)/(nPphi*nH);
    mu=mu_orb[imu_orb];
    df0g_orb=0.; 
    value=0.;
    nvp=2*f0_nvp+1;
    nmu=f0_nmu+1;
    nnode=max_node-min_node+1;
    if (idx==1){
      dvpdt=0.;
      dFdvp=0.;
      E[0]=0.; E[1]=0.; E[2]=0.;
      F_node[0]=0.; F_node[1]=0.; F_node[2]=0.;
      grad_F[0]=0.; grad_F[1]=0.;
      dxdt[0]=0.; dxdt[1]=0.; dxdt[2]=0.;
      dvpdt=0.;
      dFdvp=0.;
      dFdRphi=0.;
      if (use_ff){
        Br_l=0.; Bz_l=0.; Bphi_l=0.;
        gradParF=0.;
      }
    }
    r=rt[iorb*nt+it_orb];z=zt[iorb*nt+it_orb];
    vp=vpt[iorb*nt+it_orb];itr=itr_save[iorb*nt+it_orb];
    p[0]=p_save[iorb*nt*3+it_orb*3+0];
    p[1]=p_save[iorb*nt*3+it_orb*3+1];
    p[2]=p_save[iorb*nt*3+it_orb*3+2];
    if (itr>0){
      for(int k=0;k<3;k++){
        node=nd[k*num_tri+itr-1];
        Br=B[(node-1)*3+0];Bz=B[(node-1)*3+1];Bphi=B[(node-1)*3+2];
        Bmag=sqrt(Br*Br+Bz*Bz+Bphi*Bphi);
        tempi=Ti[node-1]*1.6022E-19;
        mu_n=2*mu*Bmag/tempi;
        vp_n=vp/sqrt(tempi/mi);
        if (idx==1){
          rho=mi*vp/qi/Bmag;
          D=1.0/(1.0+rho*nb_curl_nb[node-1]);
          if (nrho>0){
            rho_perp=min(rhomax,sqrt(2*mi*mu/Bmag)/qi);
            rhon=rho_perp/(drho+1E-10);
            irho0=min(int(floor(rhon)),nrho-1);
            irho1=irho0+1;
            wrho[1]=rhon-double(irho0);
            wrho[0]=1.0-wrho[1];
          }else{
            irho0=0;
            irho1=0;
            wrho[0]=1.0;
            wrho[1]=0.0;
          }
          E[2]=wrho[0]*Ephi[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho0]\
              +wrho[1]*Ephi[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho1];
          if (basis[node-1]==1){
            E[0]=wrho[0]*Er[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho0];
                +wrho[1]*Er[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho1];
            E[1]=wrho[0]*Ez[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho0];
                +wrho[1]*Ez[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho1];
          }
          else{
            E[0]=wrho[0]*(Er[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho0]*Bz\
                         +Ez[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho0]*Br)\
                +wrho[1]*(Er[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho1]*Bz\
                         +Ez[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho1]*Br);
            E[0]=E[0]/sqrt(Br*Br+Bz*Bz);
            E[1]=wrho[0]*(-Er[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho0]*Br\
                          +Ez[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho0]*Bz)\
                +wrho[1]*(-Er[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho1]*Br\
                          +Ez[(node-1)*nsteps_loop*(nrho+1)+istep*(nrho+1)+irho1]*Bz);
            E[1]=E[1]/sqrt(Br*Br+Bz*Bz);
          }
        dxdt[0]=dxdt[0]+p[k]*D*(Bz*E[2]-Bphi*E[1])/(Bmag*Bmag);
        dxdt[1]=dxdt[1]+p[k]*D*(Bphi*E[0]-Br*E[2])/(Bmag*Bmag);
        dxdt[2]=dxdt[2]+p[k]*D*(Br*E[1]-Bz*E[0])/(Bmag*Bmag);
        dvpdt=dvpdt+p[k]*D*qi/mi*(E[0]*(Br/Bmag+rho*curlbr[node-1])\
                                 +E[1]*(Bz/Bmag+rho*curlbz[node-1])\
                                 +E[2]*(Bphi/Bmag+rho*curlbphi[node-1]));
        }//end if idx==1
        if((mu_n<f0_smu_max*f0_smu_max)&&(vp_n<f0_vp_max)&&(vp_n>=-f0_vp_max)){
          smu=sqrt(mu_n);
          wmu[0]=smu/f0_dsmu;
          imu_f0=floor(wmu[0]);
          wmu[1]=wmu[0]-double(imu_f0);
          wmu[0]=1.0-wmu[1];
          wvp[0]=vp_n/f0_dvp;
          ivp_f0=floor(wvp[0]);
          wvp[1]=wvp[0]-double(ivp_f0);
          wvp[0]=1.0-wvp[1];
          imu=imu_f0;
          ivp=ivp_f0+f0_nvp;
          inode=node-min_node;
          tmp[0][0]=df0g[ivp*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+imu*nsteps_loop+istep];
          tmp[0][1]=df0g[ivp*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+(imu+1)*nsteps_loop+istep];
          tmp[1][0]=df0g[(ivp+1)*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+imu*nsteps_loop+istep];
          tmp[1][1]=df0g[(ivp+1)*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+(imu+1)*nsteps_loop+istep];
          value=tmp[0][0]*wvp[0]*wmu[0]+tmp[0][1]*wvp[0]*wmu[1]\
                      +tmp[1][0]*wvp[1]*wmu[0]+tmp[1][1]*wvp[1]*wmu[1];
          if (idx==1){
            F_node[k]=value/sqrt(2*mu*Bmag);
            dFdvp=dFdvp+(wmu[0]*(tmp[1][0]-tmp[0][0])+wmu[1]*(tmp[1][1]-tmp[0][1]))\
                         *p[k]/sqrt(2*mu*Bmag)/(f0_dvp*sqrt(tempi/mi));
            if ((nphi>1)&&(! use_ff)){
              tmp[0][0]=dFdphi[ivp*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+imu*nsteps_loop+istep];
              tmp[0][1]=dFdphi[ivp*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+(imu+1)*nsteps_loop+istep];
              tmp[1][0]=dFdphi[(ivp+1)*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+imu*nsteps_loop+istep];
              tmp[1][1]=dFdphi[(ivp+1)*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+(imu+1)*nsteps_loop+istep];
              value=tmp[0][0]*wvp[0]*wmu[0]+tmp[0][1]*wvp[0]*wmu[1]\
                          +tmp[1][0]*wvp[1]*wmu[0]+tmp[1][1]*wvp[1]*wmu[1];
              dFdRphi=dFdRphi+p[k]*value/sqrt(2*mu*Bmag)/rz[(node-1)*2+0];
            }
            if ((nphi>1)&&(use_ff)){
              Br_l=Br_l+p[k]*Br;
              Bz_l=Bz_l+p[k]*Bz;
              Bphi_l=Bphi_l+p[k]*Bphi;
              tmp[0][0]=dFdpara[ivp*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+imu*nsteps_loop+istep];
              tmp[0][1]=dFdpara[ivp*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+(imu+1)*nsteps_loop+istep];
              tmp[1][0]=dFdpara[(ivp+1)*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+imu*nsteps_loop+istep];
              tmp[1][1]=dFdpara[(ivp+1)*nnode*nmu*nsteps_loop+inode*nmu*nsteps_loop+(imu+1)*nsteps_loop+istep];
              value=tmp[0][0]*wvp[0]*wmu[0]+tmp[0][1]*wvp[0]*wmu[1]\
                          +tmp[1][0]*wvp[1]*wmu[0]+tmp[1][1]*wvp[1]*wmu[1];
              gradParF=gradParF+p[k]*value/sqrt(2*mu*Bmag);
            }
          }
          else{
            df0g_orb=df0g_orb+value*p[k]/sqrt(2*mu*Bmag);
          }
        }
      }//end for k
      if (idx==1){
        gradF_orb(F_node,itr,num_tri,nd,rz,grad_F);
        if ((nphi>1)&&(use_ff)){
          Bmag_l=sqrt(Br_l*Br_l+Bz_l*Bz_l+Bphi_l*Bphi_l);
          if (Bmag_l>1E-4) dFdRphi=(gradParF*Bmag_l-grad_F[0]*Br_l-grad_F[1]*Bz_l)/Bphi_l;
        }
        df0g_orb=-dxdt[0]*grad_F[0]-dxdt[1]*grad_F[1]-dxdt[2]*dFdRphi-dvpdt*dFdvp;
        df0g_orb=df0g_orb*sml_dt;
      }
    }//end if itr>0
    df0g_orb=df0g_orb*dt_orb[iorb]/sml_dt;
    dF_orb[iorb*nt*nsteps_loop+it_orb*nsteps_loop+istep]=df0g_orb*pow(mi/atan(1.)/8,1.5)/1.6022E-19;
    iorb=iorb+nblocks_max;
    }
  }
  ''','dF_orb_main')
  nblocks_max=1024
  mynorb=iorb2-iorb1+1
  nblocks=min(nblocks_max,mynorb)
  mu_orb_gpu=cp.array(orbit.mu_orb,dtype=cp.float64)
  R_orb_gpu=cp.array(orbit.R_orb,dtype=cp.float64).ravel(order='C')
  Z_orb_gpu=cp.array(orbit.Z_orb,dtype=cp.float64).ravel(order='C')
  vp_orb_gpu=cp.array(orbit.vp_orb,dtype=cp.float64).ravel(order='C')
  itr_gpu=cp.array(grid.itr_save,dtype=cp.int32).ravel(order='C')
  p_gpu=cp.array(grid.p_save,dtype=cp.float64).ravel(order='C')
  steps_orb_gpu=cp.array(orbit.steps_orb[iorb1-1:iorb2],dtype=cp.int32)
  dt_orb_gpu=cp.array(orbit.dt_orb[iorb1-1:iorb2],dtype=cp.float64)
  num_tri=np.shape(grid.nd)[1]
  dF_orb_gpu=cp.zeros((mynorb*nsteps_loop*orbit.nt,),dtype=cp.float64)
  if (idx==1)and(xgc=='xgc1')and(gyro_E):
    from parameters import nrho
  else:
    nrho=0
  if nrho>0:
    from parameters import rhomax
  else:
    rhomax=0.0
  dF_orb_kernel((nblocks*nsteps_loop,),(orbit.nt,),(dF_orb_gpu,mu_orb_gpu,int(nsteps_loop),\
     steps_orb_gpu,int(orbit.nPphi),int(orbit.nH),int(orbit.nt),int(nblocks_max),int(mynorb),\
     int(idx),R_orb_gpu,Z_orb_gpu,vp_orb_gpu,itr_gpu,p_gpu,nd_gpu,int(num_tri),B_gpu,Ti_gpu,mi,qi,\
     nb_curl_nb_gpu,curlbr_gpu,curlbz_gpu,curlbphi_gpu,basis_gpu,Er_gpu,Ez_gpu,Ephi_gpu,\
     float(grid.f0_smu_max),float(grid.f0_vp_max),float(grid.f0_dsmu),float(grid.f0_dvp),\
     int(grid.f0_nvp),int(grid.f0_nmu),int(grid.min_node),int(grid.max_node),df0g_gpu,dFdphi_gpu,\
     sml_dt,dt_orb_gpu,rz_gpu,int(iorb1),int(grid.nphi),int(nrho),float(rhomax),dFdpara_gpu,use_ff))
  dF_orb=cp.asnumpy(dF_orb_gpu).reshape((mynorb,orbit.nt,nsteps_loop),order='C')
  del mu_orb_gpu,R_orb_gpu,Z_orb_gpu,vp_orb_gpu,itr_gpu,p_gpu,steps_orb_gpu,dt_orb_gpu,dF_orb_gpu
  return np.sum(dF_orb,axis=1)

def copy_data(idx,nsteps_loop,iphi):
  import cupy as cp
  import grid,orbit
  from parameters import xgc,use_ff
  global df0g_gpu,nd_gpu,nb_curl_nb_gpu,curlbr_gpu,curlbz_gpu,curlbphi_gpu,\
         basis_gpu,Er_gpu,Ez_gpu,Ephi_gpu,B_gpu,Ti_gpu,rz_gpu,dFdphi_gpu,dFdpara_gpu
  df0g_gpu=cp.array(grid.df0g[:,:,:,iphi,:],dtype=cp.float64).ravel(order='C')
  B_gpu=cp.array(grid.B,dtype=cp.float64).ravel(order='C')
  Ti_gpu=cp.array(grid.tempi,dtype=cp.float64)
  rz_gpu=cp.array(grid.rz,dtype=cp.float64).ravel(order='C')
  nd_gpu=cp.array(grid.nd,dtype=cp.int32).ravel(order='C')
  if idx==1:
    Er_gpu=cp.array(grid.Er[iphi,:,:,:],dtype=cp.float64).ravel(order='C')
    Ez_gpu=cp.array(grid.Ez[iphi,:,:,:],dtype=cp.float64).ravel(order='C')
    Ephi_gpu=cp.array(grid.Ephi[iphi,:,:,:],dtype=cp.float64).ravel(order='C')
    nb_curl_nb_gpu=cp.array(grid.nb_curl_nb,dtype=cp.float64)
    curlbr_gpu=cp.array(grid.curlbr,dtype=cp.float64)
    curlbz_gpu=cp.array(grid.curlbz,dtype=cp.float64)
    curlbphi_gpu=cp.array(grid.curlbphi,dtype=cp.float64)
    basis_gpu=cp.array(grid.basis,dtype=cp.int32)
  else:
    Er_gpu=cp.zeros((1,),dtype=cp.float64)
    Ez_gpu=cp.zeros((1,),dtype=cp.float64)
    Ephi_gpu=cp.zeros((1,),dtype=cp.float64)
    nb_curl_nb_gpu=cp.zeros((1,),dtype=cp.float64)
    curlbr_gpu=cp.zeros((1,),dtype=cp.float64)
    curlbz_gpu=cp.zeros((1,),dtype=cp.float64)
    curlbphi_gpu=cp.zeros((1,),dtype=cp.float64)
    basis_gpu=cp.zeros((1,),dtype=cp.int32)

  if (xgc=='xgc1')and(idx==1):
    if use_ff:
      dFdphi_gpu=cp.zeros((1,),dtype=cp.float64)
      dFdpara_gpu=grid.gradParF_ff_gpu(B_gpu,df0g_gpu,iphi,nsteps_loop)
    else:
      from numpy import pi
      dphi=2*pi/float(grid.nphi*grid.nwedge)
      iphip1=(iphi+1)%grid.nphi
      iphim1=(iphi-1)%grid.nphi
      df0g_iphip1_gpu=cp.array(grid.df0g[:,:,:,iphip1,:],dtype=cp.float64).ravel(order='C')
      df0g_iphim1_gpu=cp.array(grid.df0g[:,:,:,iphim1,:],dtype=cp.float64).ravel(order='C')
      dFdphi_gpu=(df0g_iphip1_gpu-df0g_iphim1_gpu)/2/dphi
      dFdpara_gpu=cp.zeros((1,),dtype=cp.float64)
      del df0g_iphip1_gpu,df0g_iphim1_gpu
  else:
    dFdphi_gpu=cp.zeros((1,),dtype=cp.float64)
    dFdpara_gpu=cp.zeros((1,),dtype=cp.float64)
  return

def clear_data():
  import cupy
  global df0g_gpu,nd_gpu,nb_curl_nb_gpu,curlbr_gpu,curlbz_gpu,curlbphi_gpu,\
         basis_gpu,Er_gpu,Ez_gpu,Ephi_gpu,B_gpu,Ti_gpu,rz_gpu,dFdphi_gpu,dFdpara_gpu
  del df0g_gpu,nd_gpu,nb_curl_nb_gpu,curlbr_gpu,curlbz_gpu,curlbphi_gpu,\
         basis_gpu,Er_gpu,Ez_gpu,Ephi_gpu,B_gpu,Ti_gpu,rz_gpu,dFdphi_gpu,dFdpara_gpu
  mempool = cupy.get_default_memory_pool()
  pinned_mempool = cupy.get_default_pinned_memory_pool()
  mempool.free_all_blocks()
  pinned_mempool.free_all_blocks()
