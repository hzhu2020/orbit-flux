import adios2 as ad
import numpy as np
from math import floor
from scipy.interpolate import griddata

def read(xgc_dir,Nr,Nz):
  global rz,guess_table,guess_xtable,guess_count,guess_list,mapping,\
         guess_min,inv_guess_d,nnode,nd,psix,psi_rz,psi2d,rlin,zlin,R,Z,\
         B,tempi,f0_smu_max,f0_vp_max,f0_dsmu,f0_dvp,f0_nvp
  fname=xgc_dir+'/xgc.mesh.bp'
  fid=ad.open(fname,'r')
  rz=fid.read('/coordinates/values')
  psi_rz=fid.read('psi')
  nnode=np.size(psi_rz)
  guess_table=fid.read('guess_table')
  guess_xtable=fid.read('guess_xtable')
  guess_count=fid.read('guess_count')
  guess_list=fid.read('guess_list')
  mapping=fid.read('mapping')
  guess_min=fid.read('grid%guess_min')
  inv_guess_d=fid.read('grid%inv_guess_d')
  nd=fid.read('nd')
  fid.close()

  fname=xgc_dir+'/xgc.equil.bp'
  fid=ad.open(fname,'r')
  psix=fid.read('eq_x_psi')
  fid.close()

  fname=xgc_dir+'/xgc.bfield.bp'
  fid=ad.open(fname,'r')
  B=fid.read('bfield')
  fid.close()

  fname=xgc_dir+'/xgc.f0.mesh.bp'
  fid=ad.open(fname,'r')
  tempi=fid.read('f0_T_ev')
  tempi=np.squeeze(tempi)#important for dimension match
  #for kinetic-electron simulations, choose ion temperature
  if tempi.ndim>1: tempi=tempi[1,:]
  f0_smu_max=fid.read('f0_smu_max')
  f0_vp_max=fid.read('f0_vp_max')
  f0_dsmu=fid.read('f0_dsmu')
  f0_dvp=fid.read('f0_dvp')
  f0_nvp=fid.read('f0_nvp')
  fid.close()
  #XGC outputs in Fortran order but here the order is different
  guess_table=np.transpose(guess_table)
  mapping=np.transpose(mapping)
  guess_xtable=np.transpose(guess_xtable)
  guess_count=np.transpose(guess_count)
  nd=np.transpose(nd)

  rmesh=rz[:,0]
  zmesh=rz[:,1]
  rlin=np.linspace(min(rmesh),max(rmesh),Nr)
  zlin=np.linspace(min(zmesh),max(zmesh),Nz)
  R,Z=np.meshgrid(rlin,zlin)
  psi2d=griddata(rz,psi_rz,(R,Z),method='cubic')

def readf0(xgc_dir,source,idx,start_gstep,nsteps,period,min_node,max_node):
  global df0g
  for istep in range(nsteps):
    gstep=start_gstep+istep*period
    if (idx==1)or(idx==5):
      fname=xgc_dir+'/xgc.f0.'+'{:0>5d}'.format(gstep)+'.bp'
    else:
      fname=xgc_dir+'/xgc.orbit.'+source+'.'+'{:0>5d}'.format(gstep)+'.bp'
    fid=ad.open(fname,'r')
    nmu=fid.read('mudata')
    nvp=fid.read('vpdata')
    n_node=max_node-min_node+1
    if istep==0: df0g=np.zeros((nvp,n_node,nmu,nsteps),dtype=float)
    if (idx==1)or(idx==5):
      tmp=fid.read('i_f',start=[0,min_node-1,0],count=[nmu,n_node,nvp])
    else:
      tmp=fid.read('i_df0g',start=[0,min_node-1,0],count=[nmu,n_node,nvp])
    tmp=np.transpose(tmp)#[vp,node,mu] order as in Fortran XGC
    df0g[:,:,:,istep]=tmp

def grid_deriv_init(xgc_dir):
  global nelement_r,eindex_r,value_r,nelement_z,eindex_z,value_z
  fname=xgc_dir+'/xgc.grad_rz.bp'
  fid=ad.open(fname,'r')
  nelement_r=fid.read('nelement_r')
  eindex_r=fid.read('eindex_r')
  value_r=fid.read('value_r')
  nelement_z=fid.read('nelement_z')
  eindex_z=fid.read('eindex_z')
  value_z=fid.read('value_z')

  nelement_r=np.transpose(nelement_r)
  eindex_r=np.transpose(eindex_r)
  value_r=np.transpose(value_r)
  nelement_z=np.transpose(nelement_z)
  eindex_z=np.transpose(eindex_z)
  value_z=np.transpose(value_z)
  fid.close()
  
def additional_Bfield(xgc_dir,Nr,Nz): 
  global basis,nb_curl_nb,gradBr,gradBz,gradBphi,curlbr,curlbz,curlbphi
  fname=xgc_dir+'/xgc.grad_rz.bp'
  fid=ad.open(fname,'r')
  basis=fid.read('basis')
  fid.close()

  fname=xgc_dir+'/xgc.f0.mesh.bp'
  fid=ad.open(fname,'r')
  nb_curl_nb=fid.read('nb_curl_nb')
  fid.close()
  
  Br=griddata(rz,B[:,0],(R,Z),method='cubic')
  Bz=griddata(rz,B[:,1],(R,Z),method='cubic')
  Bphi=griddata(rz,B[:,2],(R,Z),method='cubic')
  Bmag=np.sqrt(Br**2+Bz**2+Bphi**2)
  
  br=Br/Bmag
  bz=Bz/Bmag
  bphi=Bphi/Bmag
  curlbr2d,curlbz2d,curlbphi2d=Curl(rlin,zlin,br,bz,bphi,Nr,Nz)
  curlbr=np.zeros((nnode),dtype=float)
  curlbz=np.zeros((nnode),dtype=float)
  for i in range(nnode):
    curlbr[i]=TwoD(R,Z,curlbr2d,rz[i,0],rz[i,1])
    curlbz[i]=TwoD(R,Z,curlbz2d,rz[i,0],rz[i,1])

def read_dpot_orb(orbit_dir):
  global dpot_orb
  dpot_orb=np.zeros((nnode))
  fname=orbit_dir+'/pot0m.txt'
  fid=open(fname,'r')
  dum=int(fid.readline(8))
  if dum!=nnode:
    print('Wrong nnode for pot0m.txt. Set dpot_orb=0.',flush=True)
    return
  fid.readline(8)
  for i in range(nnode):
    value=fid.readline(19)
    dpot_orb[i]=float(value)
    fid.readline(1)
  dum=int(fid.readline(8))
  if dum!=-1:
    print('Wrong file end for pot0m.txt. Set dpot_orb=0.',flush=True)
    dpot_orb[:]=0.0
    return

  fid.close()
  return

def Eturb(xgc_dir,start_gstep,nsteps,period,grad_psitheta,psi_only,min_node,max_node):
  from parameters import Eturb_pot0,Eturb_dpot 
  Er=np.zeros((nnode,nsteps),dtype=float)
  Ez=np.zeros((nnode,nsteps),dtype=float)
  for istep in range(nsteps):
    gstep=start_gstep+istep*period
    fname=xgc_dir+'/xgc.2d.'+'{:0>5d}'.format(gstep)+'.bp'
    fid=ad.open(fname,'r')
    dpot=fid.read('dpot')
    pot0=fid.read('pot0')
    dpot_turb=float(Eturb_pot0)*pot0+float(Eturb_dpot)*dpot-dpot_orb
    for i in range(min_node-1,max_node):
      for j in range(nelement_r[i]):
        ind=eindex_r[j,i]
        Er[i,istep]=Er[i,istep]+dpot_turb[ind-1]*value_r[j,i]
      for j in range(nelement_z[i]):
        ind=eindex_z[j,i]
        Ez[i,istep]=Ez[i,istep]+dpot_turb[ind-1]*value_z[j,i]
    if grad_psitheta:
      for i in range(min_node-1,max_node):
        if (basis[i]==0)and(psi_only): Ez[i,istep]==0
  Er=-Er
  Ez=-Ez
  return Er,Ez
 
def search_tr2(xy):
   itr=-1
   p=np.zeros((3,),dtype=float)
   eps=1e-10
 
   ilo,jlo=1,1
   ihi,jhi=np.shape(guess_table)
   ij=np.zeros((2,),dtype=int)
   ij[0]=floor((xy[0]-guess_min[0])*inv_guess_d[0])+1
   ij[1]=floor((xy[1]-guess_min[1])*inv_guess_d[1])+1
   i=max(ilo,min(ihi,ij[0]))
   j=max(jlo,min(jhi,ij[1]))

   istart=guess_xtable[i-1,j-1]
   iend=istart+guess_count[i-1,j-1]-1
   for k in range(istart,iend+1):
     itrig=guess_list[k-1]
     dx=xy-mapping[:,2,itrig-1]
     p[0:2]=mapping[0:2,0,itrig-1]*dx[0]+mapping[0:2,1,itrig-1]*dx[1]
     p[2]=1.0-p[0]-p[1]
     if min(p)>=-eps:
       itr=itrig
       break

   return itr,p

def t_coeff_mod(xy,itr,p):
  eps1=1e-4*psix
  psi=np.zeros((4,),dtype=float)
  psi[3]=TwoD(R,Z,psi2d,xy[0],xy[1])
  for i in range(3): psi[i]=psi_rz[nd[i,itr-1]-1]
  psi_diff=0
  if abs(psi[0]-psi[1])<=eps1:
    psi_diff=3
  elif abs(psi[0]-psi[2])<=eps1:
    psi_diff=2
  elif abs(psi[1]-psi[2])<=eps1:
    psi_diff=1

  if (psi_diff>0):
    a=psi_diff%3+1
    b=(psi_diff+1)%3+1
    psi[:]=abs(psi[:]-psi[psi_diff-1])
    p_temp=psi[3]/psi[a-1]
    p_temp=min(p_temp,1.0)
    p[psi_diff-1]=1.0-p_temp
    t_temp=p[a-1]+p[b-1]
    p[a-1]=p_temp*p[a-1]/t_temp
    p[b-1]=p_temp*p[b-1]/t_temp
  return p 

def Grad(r,z,fld,Nr,Nz):
  dr=r[2]-r[1]
  dz=z[2]-z[1]
  gradr=np.nan*np.zeros((Nz,Nr),dtype=float)
  gradz=np.nan*np.zeros((Nz,Nr),dtype=float)
  gradphi=np.nan*np.zeros((Nz,Nr),dtype=float)
  for i in range(1,Nz-1):
    for j in range(1,Nr-1):
      gradr[i,j]=(fld[i,j+1]-fld[i,j-1])/2/dr
      gradz[i,j]=(fld[i+1,j]-fld[i-1,j])/2/dz
      gradphi[i,j]=0.0 #assuming axisymmetry
  return gradr,gradz,gradphi

def Curl(r,z,fldr,fldz,fldphi,Nr,Nz):
  #all calculations below assume axisymmetry
  dr=r[2]-r[1]
  dz=z[2]-z[1]
  curlr=np.nan*np.zeros((Nz,Nr),dtype=float)
  curlz=np.nan*np.zeros((Nz,Nr),dtype=float)
  curlphi=np.nan*np.zeros((Nz,Nr),dtype=float)
  for i in range(1,Nz-1):
    for j in range(1,Nr-1):
      curlr[i,j]=-(fldphi[i+1,j]-fldphi[i-1,j])/2/dz
      curlphi[i,j]=(fldr[i+1,j]-fldr[i-1,j])/2/dz-(fldz[i,j+1]-fldz[i,j-1])/2/dr
      curlz[i,j]=(r[j+1]*fldphi[i,j+1]-r[j-1]*fldphi[i,j-1])/2/dr/r[j]
  return curlr,curlz,curlphi

def TwoD(x2d,y2d,f2d,xin,yin):
  Ny,Nx=np.shape(x2d)
  x0=x2d[0,0]
  dx=x2d[0,1]-x0
  y0=y2d[0,0]
  dy=y2d[1,0]-y0
  ix=floor((xin-x0)/dx)
  wx=(xin-x0)/dx-ix
  iy=floor((yin-y0)/dy)
  wy=(yin-y0)/dy-iy
  if (ix<0) or (ix>Nx-2) or (iy<0) or (iy>Ny-2):
    fout=np.nan
  else:
    fout=f2d[iy,ix]*(1-wy)*(1-wx) + f2d[iy+1,ix]*wy*(1-wx)\
        +f2d[iy,ix+1]*(1-wy)*wx + f2d[iy+1,ix+1]*wy*wx

  return fout

def gradF_orb(F,itr,nsteps):
  dFdx=np.zeros((2,nsteps),dtype=float)
  r=np.zeros((3,),dtype=float)
  z=np.zeros((3,),dtype=float)
  for i in range(3):
    node=nd[i,itr-1]
    r[i]=rz[node-1,0]
    z[i]=rz[node-1,1]

  xsj=r[0]*(z[1]-z[2])+r[1]*(z[2]-z[0])+r[2]*(z[0]-z[1])
  if (xsj==0.0):
    print('Error: diag_orbit_loss at gradF_orb, xsj==0',flush=True)
    exit()
    
  dFdx[0,:]=F[0,:]*(z[1]-z[2])+F[1,:]*(z[2]-z[0])+F[2,:]*(z[0]-z[1])
  dFdx[0,:]=dFdx[0,:]/xsj
  dFdx[1,:]=F[0,:]*(r[2]-r[1])+F[1,:]*(r[0]-r[2])+F[2,:]*(r[1]-r[0])
  dFdx[1,:]=dFdx[1,:]/xsj
  return dFdx

def node_range(tri_psi):
  from orbit import iorb1,iorb2,nt,steps_orb,R_orb,Z_orb
  min_node=nnode
  max_node=0
  mynorb=iorb2-iorb1+1
  itr_save=np.zeros((mynorb,nt),dtype=int)
  p_save=np.zeros((mynorb,nt,3),dtype=float)
  for iorb in range(iorb1,iorb2+1):
    for it_orb in range(steps_orb[iorb-1]):
      r=R_orb[iorb-iorb1,it_orb]
      z=Z_orb[iorb-iorb1,it_orb]
      itr,p=search_tr2([r,z])
      if (tri_psi)and(itr>0)and(max(p)<1.0): p=t_coeff_mod([r,z],itr,p)
      itr_save[iorb-iorb1,it_orb]=itr
      p_save[iorb-iorb1,it_orb,:]=p[:]
      if itr>0:
        for i in range(3):
          node=nd[i,itr-1]
          if (node>max_node): max_node=node
          if (node<min_node): min_node=node
  return min_node,max_node,itr_save,p_save

def node_range_gpu(tri_psi):
  import cupy as cp
  node_range_kernel = cp.RawKernel(r'''
  extern "C" __device__    
  void search_tr2(double r,double z,int* itr,double* p,int ihi,int jhi,int num_tri,double* guess_min,\
    double* inv_guess_d,int* guess_xtable,int* guess_list,int* guess_count,double* mapping)
  {
    int ilo,jlo,ij[2],ix,iy,istart,iend,itrig;
    double eps=1E-10,dx,dy,tmp;    
    itr[0]=-1;
    p[0]=0.;
    p[1]=0.;
    p[2]=0.;
    ilo=1;
    jlo=1;
    ij[0]=floor((r-guess_min[0])*inv_guess_d[0])+1;
    ij[1]=floor((z-guess_min[1])*inv_guess_d[1])+1;
    ix=min(int(ihi),ij[0]);
    ix=max(ilo,ix);
    iy=min(int(jhi),ij[1]);
    iy=max(jlo,iy);
    istart=guess_xtable[(ix-1)*jhi+iy-1];
    iend=istart+guess_count[(ix-1)*jhi+iy-1]-1;
    for (int k=istart;k<=iend;k++){
      itrig=guess_list[k-1];
      dx=r-mapping[0*3*num_tri+2*num_tri+itrig-1];
      dy=z-mapping[1*3*num_tri+2*num_tri+itrig-1];   
      p[0]=mapping[0*3*num_tri+0*num_tri+itrig-1]*dx\
          +mapping[0*3*num_tri+1*num_tri+itrig-1]*dy;
      p[1]=mapping[1*3*num_tri+0*num_tri+itrig-1]*dx\
          +mapping[1*3*num_tri+1*num_tri+itrig-1]*dy;
      p[2]=1.-p[0]-p[1];
      tmp=min(p[0],p[1]);
      tmp=min(tmp,p[2]);
      if (tmp>=-eps){
        itr[0]=itrig;
        break;
      }
    }
  }
  extern "C" __device__ 
  void t_coeff_mod(double r,double z,double* rlin,double* zlin,double* p,double psix,\
    double* psi2d,double* psi_rz,int itr,int num_tri,int* nd,int Nr,int Nz)
  {
    int ir,iz,psi_diff,a,b;
    double wr,wz,psi[4],tmp,eps1,p_temp,t_temp,r0,z0,dr,dz;
    r0=rlin[0];
    z0=zlin[0];
    dr=rlin[1]-rlin[0];
    dz=zlin[1]-zlin[0];
    eps1=1E-4*psix;
    ir=floor((r-r0)/dr);
    iz=floor((z-z0)/dz);
    wr=(r-r0)/dr-ir;
    wz=(z-z0)/dz-iz;
    psi[0]=psi_rz[nd[0*num_tri+itr-1]-1];
    psi[1]=psi_rz[nd[1*num_tri+itr-1]-1];
    psi[2]=psi_rz[nd[2*num_tri+itr-1]-1];
    psi[3]=psi2d[iz*Nr+ir]*(1-wz)*(1-wr)+psi2d[(iz+1)*Nr+ir]*wz*(1-wr)\
          +psi2d[iz*Nr+ir+1]*(1-wz)*wr+psi2d[(iz+1)*Nr+ir+1]*wz*wr;
    psi_diff=0;
    if (abs(psi[0]-psi[1])<=eps1){
      psi_diff=3;
    }
    else if (abs(psi[0]-psi[2])<=eps1){
      psi_diff=2;
    }
    else if (abs(psi[1]-psi[2])<=eps1){
      psi_diff=1;
    }
    if (psi_diff>0)
    {
      a=psi_diff%3+1;
      b=(psi_diff+1)%3+1;
      tmp=psi[psi_diff-1];
      psi[0]=abs(psi[0]-tmp);
      psi[1]=abs(psi[1]-tmp);
      psi[2]=abs(psi[2]-tmp);
      psi[3]=abs(psi[3]-tmp);
      p_temp=psi[3]/psi[a-1];
      p_temp=min(p_temp,1.0);
      p[psi_diff-1]=1.0-p_temp;
      t_temp=p[a-1]+p[b-1];
      p[a-1]=p_temp*p[a-1]/t_temp;
      p[b-1]=p_temp*p[b-1]/t_temp;
    }
  }   
  extern "C" __global__
  void node_range(double* rt,double* zt,int steps_orb,int nt,int* itr_save,double* p_save,\
      int ihi,int jhi,int num_tri,double* guess_min,double* inv_guess_d,int* guess_xtable,int* guess_list,\
      int* guess_count,double* mapping,bool tri_psi,double* rlin,double* zlin,double psix,double* psi2d,\
      double* psi_rz,int* nd,int Nr,int Nz) 
  {
      double r,z,p[3],tmp; 
      int it_orb,itr[1],ilo,jlo,ij[2],ix,iy,istart,iend; 
      it_orb=blockDim.x*blockIdx.x+threadIdx.x;
      if (it_orb>=steps_orb) return;
      r=rt[it_orb];
      z=zt[it_orb];
      search_tr2(r,z,itr,p,ihi,jhi,num_tri,guess_min,inv_guess_d,guess_xtable,guess_list,guess_count,mapping);
      tmp=max(p[0],p[1]);
      tmp=max(tmp,p[2]);
      if ((tri_psi)&&(itr[0]>0)&&(tmp<1.0))
      {
        t_coeff_mod(r,z,rlin,zlin,p,psix,psi2d,psi_rz,itr[0],num_tri,nd,Nr,Nz);
      }
      itr_save[it_orb]=itr[0];
      p_save[it_orb*3+0]=p[0];
      p_save[it_orb*3+1]=p[1];
      p_save[it_orb*3+2]=p[2]; 
  }
  ''', 'node_range')
  from orbit import iorb1,iorb2,nt,steps_orb,R_orb,Z_orb
  min_node=nnode
  max_node=0
  mynorb=iorb2-iorb1+1
  itr_save=np.zeros((mynorb,nt),dtype=int)
  p_save=np.zeros((mynorb,nt,3),dtype=float)
  num_tri=np.shape(mapping)[2]
  ihi,jhi=np.shape(guess_table)
  guess_min_gpu=cp.array(guess_min,dtype=cp.float64)
  inv_guess_d_gpu=cp.array(inv_guess_d,dtype=cp.float64)
  guess_xtable_gpu=cp.array(guess_xtable,dtype=cp.int32).ravel(order='C')
  guess_list_gpu=cp.array(guess_list,dtype=cp.int32)
  guess_count_gpu=cp.array(guess_count,dtype=cp.int32).ravel(order='C')
  mapping_gpu=cp.array(mapping,dtype=cp.float64).ravel(order='C')
  nd_gpu=cp.array(nd,dtype=cp.int32).ravel(order='C')
  rlin_gpu=cp.array(rlin,dtype=cp.float64)
  zlin_gpu=cp.array(zlin,dtype=cp.float64)
  psi2d_gpu=cp.array(psi2d,dtype=cp.float64).ravel(order='C')
  psi_rz_gpu=cp.array(psi_rz,dtype=cp.float64)
  for iorb in range(iorb1,iorb2+1):
    rt_gpu=cp.asarray(R_orb[iorb-iorb1,:],dtype=cp.float64)
    zt_gpu=cp.asarray(Z_orb[iorb-iorb1,:],dtype=cp.float64)
    itr_save_gpu=cp.zeros((nt,),dtype=cp.int32)
    p_save_gpu=cp.zeros((nt*3,),dtype=cp.float64)
    node_range_kernel((nt,),(1,),(rt_gpu,zt_gpu,steps_orb[iorb-1],int(nt),itr_save_gpu,p_save_gpu,\
      int(ihi),int(jhi),int(num_tri),guess_min_gpu,inv_guess_d_gpu,guess_xtable_gpu,guess_list_gpu,\
      guess_count_gpu,mapping_gpu,tri_psi,rlin_gpu,zlin_gpu,float(psix),psi2d_gpu,psi_rz_gpu,\
      nd_gpu,int(rlin.size),int(zlin.size)))
    itr_save[iorb-iorb1,:]=cp.asnumpy(itr_save_gpu)
    p_save[iorb-iorb1,:,:]=cp.asnumpy(p_save_gpu).reshape(nt,3,order='C')
    for it_orb in range(steps_orb[iorb-1]):
      itr=itr_save[iorb-iorb1,it_orb]
      if itr>0:
        for i in range(3):
          node=nd[i,itr-1]
          if (node>max_node): max_node=node
          if (node<min_node): min_node=node

  return min_node,max_node,itr_save,p_save
