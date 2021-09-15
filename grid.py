import adios2 as ad
import numpy as np
from math import floor
from scipy.interpolate import griddata

def read(xgc,use_ff,xgc_dir,Nr,Nz):
  global rz,guess_table,guess_xtable,guess_count,guess_list,mapping,\
         guess_min,inv_guess_d,nnode,nd,psix,psi_rz,psi2d,rlin,zlin,R,Z,\
         B,tempi,f0_smu_max,f0_vp_max,f0_dsmu,f0_dvp,f0_nvp,f0_nmu
  fname=xgc_dir+'/xgc.mesh.bp'
  fid=ad.open(fname,'r')
  if xgc=='xgca':
    rz=fid.read('/coordinates/values')
  elif xgc=='xgc1':
    rz=fid.read('rz')
    if use_ff:
      global ff_1dp_tr,ff_1dp_p,ff_1dp_dx
      ff_1dp_tr=fid.read('ff_1dp_tr')
      ff_1dp_p=fid.read('ff_1dp_p')
      ff_1dp_dx=fid.read('one_per_dx')
  else:
    print('Wrong parameter xgc=',xgc)
  guess_min=fid.read('guess_min')
  inv_guess_d=fid.read('inv_guess_d')
  psi_rz=fid.read('psi')
  nnode=np.size(psi_rz)
  guess_table=fid.read('guess_table')
  guess_xtable=fid.read('guess_xtable')
  guess_count=fid.read('guess_count')
  guess_list=fid.read('guess_list')
  mapping=fid.read('mapping')
  nd=fid.read('nd')
  fid.close()

  fname=xgc_dir+'/xgc.equil.bp'
  fid=ad.open(fname,'r')
  psix=fid.read('eq_x_psi')
  fid.close()

  fname=xgc_dir+'/xgc.bfield.bp'
  fid=ad.open(fname,'r')
  if xgc=='xgca':
    B=fid.read('bfield')
  elif xgc=='xgc1':
    B=fid.read('/node_data[0]/values')
  else:
    print('Wrong parameter xgc=',xgc)
  fid.close()

  if xgc=='xgc1':
    global nwedge
    fname=xgc_dir+'/xgc.units.bp'
    fid=ad.open(fname,'r')
    nwedge=fid.read('sml_wedge_n')
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
  f0_nmu=fid.read('f0_nmu')
  fid.close()
  #XGC outputs in Fortran order but here the order is different
  guess_table=np.transpose(guess_table)
  mapping=np.transpose(mapping)
  guess_xtable=np.transpose(guess_xtable)
  guess_count=np.transpose(guess_count)
  nd=np.transpose(nd)
  if (xgc=='xgc1')and(use_ff):
    ff_1dp_tr=np.transpose(ff_1dp_tr)
    ff_1dp_p=np.transpose(ff_1dp_p)
    ff_1dp_dx=np.transpose(ff_1dp_dx)

  rmesh=rz[:,0]
  zmesh=rz[:,1]
  rlin=np.linspace(min(rmesh),max(rmesh),Nr)
  zlin=np.linspace(min(zmesh),max(zmesh),Nz)
  R,Z=np.meshgrid(rlin,zlin)
  psi2d=griddata(rz,psi_rz,(R,Z),method='cubic')
  return

def readf0(xgc,xgc_dir,source,idx,start_gstep,nsteps,period):
  global df0g,nphi
  for istep in range(nsteps):
    gstep=start_gstep+istep*period
    if (idx==1)or(idx==5):
      if xgc=='xgca':
        fname=xgc_dir+'/xgc.f0.'+'{:0>5d}'.format(gstep)+'.bp'
      elif xgc=='xgc1':
        fname=xgc_dir+'/xgc.orbit.f0.'+'{:0>5d}'.format(gstep)+'.bp'
      else:
        print('Wrong parameter xgc=',xgc)
    else:
      fname=xgc_dir+'/xgc.orbit.'+source+'.'+'{:0>5d}'.format(gstep)+'.bp'
    fid=ad.open(fname,'r')
    nmu=fid.read('mudata')
    nvp=fid.read('vpdata')
    if xgc=='xgc1':
      nphi=fid.read('nphi')
    elif xgc=='xgca':
      nphi=1

    n_node=max_node-min_node+1
    if (idx==1)or(idx==5):
      if xgc=='xgca':
        tmp=fid.read('i_f',start=[0,min_node-1,0],count=[nmu,n_node,nvp])
        tmp=np.expand_dims(tmp,axis=0)#add a dimension for nphi=1
      elif xgc=='xgc1':
        tmp=fid.read('i_f',start=[0,0,min_node-1,0],count=[nphi,nmu,n_node,nvp])
    else:
      if xgc=='xgca':
        tmp=fid.read('i_df0g',start=[0,min_node-1,0],count=[nmu,n_node,nvp])
        tmp=np.expand_dims(tmp,axis=0)
      elif xgc=='xgc1':
        tmp=fid.read('i_df0g',start=[0,0,min_node-1,0],count=[nphi,nmu,n_node,nvp])
    #apply toroidal average here, unless for turbulence flux
    if (xgc=='xgc1')and(idx!=1):
      nphi=1
      tmp=np.mean(tmp,axis=0,keepdims=True)
    tmp=np.transpose(tmp)#[vp,node,mu] order as in Fortran XGC
    if istep==0: df0g=np.zeros((nvp,n_node,nmu,nphi,nsteps),dtype=float)
    df0g[:,:,:,:,istep]=tmp
    fid.close()
  #end for istep
  return

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
  return
  
def additional_Bfield(xgc,xgc_dir,Nr,Nz):
  global basis,nb_curl_nb,curlbr,curlbz,curlbphi
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
  curlbphi=np.zeros((nnode),dtype=float)
  for i in range(nnode):
    curlbr[i]=TwoD(R,Z,curlbr2d,rz[i,0],rz[i,1])
    curlbz[i]=TwoD(R,Z,curlbz2d,rz[i,0],rz[i,1])
    if xgc=='xgc1': curlbphi[i]=TwoD(R,Z,curlbphi2d,rz[i,0],rz[i,1])

  return

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

def read_dpot_turb(xgc,xgc_dir,start_gstep,nsteps,period):
  from parameters import Eturb_pot0,Eturb_dpot
  global dpot_turb
  pot0=np.zeros((nphi,nnode),dtype=float)
  dpot=np.zeros((nphi,nnode),dtype=float)
  dpot_turb=np.zeros((nphi,nnode,nsteps),dtype=float)
  for istep in range(nsteps):
    gstep=start_gstep+istep*period
    if xgc=='xgca':
      fname=xgc_dir+'/xgc.2d.'+'{:0>5d}'.format(gstep)+'.bp'
    elif xgc=='xgc1':
      fname=xgc_dir+'/xgc.3d.'+'{:0>5d}'.format(gstep)+'.bp'
    fid=ad.open(fname,'r')
    dpot[:,:]=fid.read('dpot')
    pot0[:,:]=fid.read('pot0')
    if xgc=='xgc1':#move n=0,m!=0 part to pot0
      pot0=pot0+np.tile(np.mean(dpot,axis=0),(nphi,1))
      dpot=dpot-np.tile(np.mean(dpot,axis=0),(nphi,1))
    dpot_turb[:,:,istep]=float(Eturb_pot0)*pot0+float(Eturb_dpot)*dpot-np.tile(dpot_orb,(nphi,1))
    fid.close() 

  return

def gyropot(use_gpu,nsteps,itask1,itask2):
  from parameters import nrho,rhomax,ngyro
  global dpot_turb_rho
  dpot_turb_rho=np.zeros((nphi,nnode,nsteps,nrho))
  istep_old=-1
  iphi_old=-1
  for itask in range(itask1,itask2+1):
    istep=int(itask/(nphi*nrho))
    iphi=int((itask-istep*nphi*nrho)/nrho)
    irho=itask-iphi*nrho-istep*nphi*nrho
    if (istep!=istep_old)or(iphi!=iphi_old):
      dpot_2d=griddata(rz,dpot_turb[iphi,:,istep],(R,Z),method='cubic')
      #dpot_2d=cnvt_node_to_2d(dpot_turb[iphi,:,istep])
      istep_old=istep
      iphi_old=iphi
    rho=float(irho+1)*rhomax/float(nrho)
    if use_gpu:
      dpot_turb_rho[iphi,:,istep,irho]=gyropot_gpu(dpot_2d,rho,ngyro)
    else:
      dpot_turb_rho[iphi,:,istep,irho]=gyropot_cpu(dpot_2d,rho,ngyro)
  return 

#This is slower than griddata but can be easily converted to GPU
def cnvt_node_to_2d(fnode):
  from parameters import Nr,Nz,sml_tri_psi_weighting
  f2d=np.zeros((Nz,Nr),dtype=float)
  for ir in range(Nr):
    for iz in range(Nz):
      r=rlin[ir]
      z=zlin[iz]
      itr,p=search_tr2([r,z])
      if (sml_tri_psi_weighting)and(itr>0)and(max(p)<1.0): p=t_coeff_mod([r,z],itr,p)
      if itr>0:
        for i in range(3):
          node=nd[i,itr-1]
          f2d[iz,ir]=f2d[iz,ir]+p[i]*fnode[node]
      else:
        f2d[iz,ir]=np.nan
  return f2d

def gyropot_cpu(dpot2d,rho,ngyro):
  from math import floor
  dpot_rho=np.zeros((nnode,),dtype=float)
  dr=rlin[1]-rlin[0]
  dz=zlin[1]-zlin[0]
  r0=rlin[0]
  z0=zlin[0]
  Nr=np.size(rlin)
  Nz=np.size(zlin)
  for inode in range(deriv_min_node-1,deriv_max_node):
    r=rz[inode,0]
    z=rz[inode,1]
    tmp=0.0
    for igyro in range(ngyro):
      angle=2*np.pi*float(igyro)/float(ngyro)
      r1=r+rho*np.cos(angle)
      z1=z+rho*np.sin(angle)
      ir1=floor((r1-r0)/dr)
      wr=(r1-r0)/dr-ir1
      iz1=floor((z1-z0)/dz)
      wz=(z1-z0)/dz-iz1
      if (ir1<0) or (ir1>Nr-2) or (iz1<0) or (iz1>Nz-2):
        tmp=np.nan
        dpot_rho[inode]=np.nan
      else:
        tmp=dpot2d[iz1,ir1]*(1-wz)*(1-wr) + dpot2d[iz1+1,ir1]*wz*(1-wr)\
            +dpot2d[iz1,ir1+1]*(1-wz)*wr + dpot2d[iz1+1,ir1+1]*wz*wr
        dpot_rho[inode]=dpot_rho[inode]+tmp/float(ngyro)

  return dpot_rho

def gyropot_gpu(dpot2d,rho,ngyro):
  import cupy as cp
  gyropot_kernel=cp.RawKernel(r'''
  extern "C" __global__
  void gyropot(double* dpot2d,double* dpot_rho,int Nz,int Nr,int ngyro,double z0,double r0,\
                double dz,double dr,double rho,int min_node,int max_node,double* rz)
  {
    int inode,igyro,ir1,iz1;
    double r,z,angle,r1,z1,tmp,wr,wz;
    inode=blockIdx.x+min_node-1;
    while (inode<max_node){
      r=rz[inode*2+0];
      z=rz[inode*2+1];
      dpot_rho[inode]=0.0;
      for (igyro=0;igyro<ngyro;igyro++){
        angle=8.*atan(1.)*double(igyro)/double(ngyro);
        r1=r+rho*cos(angle);
        z1=z+rho*sin(angle);
        ir1=floor((r1-r0)/dr);
        iz1=floor((z1-z0)/dz);
        wr=(r1-r0)/dr-ir1;
        wz=(z1-z0)/dz-iz1;
        if ((ir1<0)||(ir1>Nr-2)||(iz1<0)||(iz1>Nz-2)){
          tmp=nan("");
        }else{
          tmp=dpot2d[iz1*Nr+ir1]*(1-wz)*(1-wr)+dpot2d[(iz1+1)*Nr+ir1]*wz*(1-wr)\
              +dpot2d[iz1*Nr+ir1+1]*(1-wz)*wr+dpot2d[(iz1+1)*Nr+ir1+1]*wz*wr;
        }
        dpot_rho[inode]+=tmp/double(ngyro);
      }
      inode=inode+gridDim.x;
    }
  }
  ''','gyropot')
  nblocks_max=4096
  nnode_deriv=deriv_max_node-deriv_min_node+1
  nblocks=min(nblocks_max,nnode_deriv)
  dpot_rho_gpu=cp.zeros((nnode,),dtype=cp.float64)
  dpot2d_gpu=cp.array(dpot2d,dtype=cp.float64).ravel(order='C')
  rz_gpu=cp.array(rz,dtype=cp.float64).ravel(order='C')
  dr=rlin[1]-rlin[0]
  dz=zlin[1]-zlin[0]
  r0=rlin[0]
  z0=zlin[0]
  Nr=np.size(rlin)
  Nz=np.size(zlin)
  gyropot_kernel((nblocks,),(1,),(dpot2d_gpu,dpot_rho_gpu,int(Nz),int(Nr),int(ngyro),float(z0),float(r0),\
                      float(dz),float(dr),float(rho),int(deriv_min_node),int(deriv_max_node),rz_gpu))
  dpot_rho=cp.asnumpy(dpot_rho_gpu)
  del dpot_rho_gpu,dpot2d_gpu,rz_gpu 
  mempool = cp.get_default_memory_pool()
  pinned_mempool = cp.get_default_pinned_memory_pool()
  mempool.free_all_blocks()
  pinned_mempool.free_all_blocks()
  return dpot_rho
 
def Eturb(xgc,use_ff,gyro_E,nsteps,grad_psitheta,psi_only):
  global Er,Ez,Ephi,nrho
  if (xgc=='xgc1')and(gyro_E):
    from parameters import nrho
  else:
    nrho=0
  Er=np.zeros((nphi,nnode,nsteps,nrho+1),dtype=float)
  Ez=np.zeros((nphi,nnode,nsteps,nrho+1),dtype=float)
  Ephi=np.zeros((nphi,nnode,nsteps,nrho+1),dtype=float)
  for i in range(min_node-1,max_node):
    for j in range(nelement_r[i]):
      ind=eindex_r[j,i]
      Er[:,i,:,0]=Er[:,i,:,0]+dpot_turb[:,ind-1,:]*value_r[j,i]
      if nrho>0: Er[:,i,:,1:nrho+1]=Er[:,i,:,1:nrho+1]+dpot_turb_rho[:,ind-1,:,:]*value_r[j,i]
    for j in range(nelement_z[i]):
      ind=eindex_z[j,i]
      Ez[:,i,:,0]=Ez[:,i,:,0]+dpot_turb[:,ind-1,:]*value_z[j,i]
      if nrho>0: Ez[:,i,:,1:nrho+1]=Ez[:,i,:,1:nrho+1]+dpot_turb_rho[:,ind-1,:,:]*value_z[j,i]
  if grad_psitheta:
    for i in range(min_node-1,max_node):
      if (basis[i]==0)and(psi_only): Ez[:,i,:,:]=0
  if (xgc=='xgc1')and(use_ff):
    Epara=np.zeros((nsteps,nrho+1),dtype=float)
    for i in range(min_node-1,max_node):
      Br=B[i,0]
      Bz=B[i,1]
      Bphi=B[i,2]
      Bmag=np.sqrt(Br**2+Bz**2+Bphi**2)
      Bpol=np.sqrt(Br**2+Bz**2)
      if Bphi>0:
        sgn=+1
      else:
        sgn=-1
      itr_l=ff_1dp_tr[i,0]
      itr_r=ff_1dp_tr[i,1]
      if (itr_l<0)or(itr_r<0): continue
      p_l=ff_1dp_p[:,i,0]
      p_r=ff_1dp_p[:,i,1]
      dl_l=ff_1dp_dx[i,0]
      dl_r=ff_1dp_dx[i,1]
      dl_tot=dl_l+dl_r
      pot_l=np.zeros((nphi,nsteps,nrho+1),dtype=float)
      pot_r=np.zeros((nphi,nsteps,nrho+1),dtype=float)
      pot_m=np.zeros((nphi,nsteps,nrho+1),dtype=float)
      pot_m[:,:,0]=dpot_turb[:,i,:]
      if nrho>0: pot_m[:,:,1:nrho+1]=dpot_turb_rho[:,i,:,:]
      for k in range(3):
        node_l=nd[k,itr_l-1]
        node_r=nd[k,itr_r-1]
        pot_l[:,:,0]=pot_l[:,:,0]+p_l[k]*dpot_turb[:,node_l-1,:]
        if nrho>0: pot_l[:,:,1:nrho+1]=pot_l[:,:,1:nrho+1]+p_l[k]*dpot_turb_rho[:,node_l-1,:,:]
        pot_r[:,:,0]=pot_r[:,:,0]+p_r[k]*dpot_turb[:,node_r-1,:]
        if nrho>0: pot_r[:,:,1:nrho+1]=pot_r[:,:,1:nrho+1]+p_r[k]*dpot_turb_rho[:,node_r-1,:,:]
      for iphi in range(nphi):
        iphip1=(iphi+1)%nphi
        iphim1=(iphi-1)%nphi
        Epara[:,:]=sgn*(-dl_r/(dl_l*dl_tot)*pot_l[iphim1,:,:]\
                   +(dl_r-dl_l)/(dl_l*dl_r)*pot_m[iphi,:,:]\
                   +dl_l/(dl_r*dl_tot)*pot_r[iphip1,:,:])
        if basis[i]==1:
          Ephi[iphi,i,:,:]=(Epara[:,:]*Bmag-Er[iphi,i,:,:]*Br-Ez[iphi,i,:,:]*Bz)/Bphi
        else:
          Ephi[iphi,i,:,:]=(Epara[:,:]*Bmag-Ez[iphi,i,:,:]*Bpol)/Bphi

  if (xgc=='xgc1')and(not use_ff):
    rz_arr=np.zeros((1,max_node-min_node+1,1))
    rz_arr[0,:,0]=rz[min_node-1:max_node,0]
    rz_arr=np.tile(rz_arr,(1,1,nsteps))
    dphi=2*np.pi/float(nphi*nwedge)
    for iphi in range(nphi):
      iphip1=(iphi+1)%nphi
      iphim1=(iphi-1)%nphi
      Ephi[iphi,min_node-1:max_node,:,0]=(dpot_turb[iphip1,min_node-1:max_node,:]\
                      -dpot_turb[iphim1,min_node-1:max_node,:])/2/dphi/rz_arr
      if nrho>0:
        rz_arr_rho=np.expand_dims(rz_arr,axis=3)
        rz_arr_rho=np.tile(rz_arr_rho,(1,1,1,nrho))
        Ephi[iphi,min_node-1:max_node,:,1:nrho+1]=(dpot_turb_rho[iphip1,min_node-1:max_node,:,:]\
                        -dpot_turb_rho[iphim1,min_node-1:max_node,:,:])/2/dphi/rz_arr_rho
  Er=-Er
  Ez=-Ez
  if xgc=='xgc1': Ephi=-Ephi
  return
 
def Eturb_gpu(xgc,gyro_E,nsteps,grad_psitheta,psi_only):
  import cupy as cp
  grid_deriv_kernel=cp.RawKernel(r'''
  extern "C" __global__
  void grid_deriv(double* dpot_turb,double* Er, double* Ez,int min_node,int max_node,int nnode,\
    int* nelement_r,int* nelement_z,int* eindex_r,int* eindex_z,double* value_r,double* value_z,
    bool grad_psitheta,bool psi_only,int* basis,double* Ephi,int nphi)
  {
    int inode,iphi,ind;
    int iphip1,iphim1;
    inode=blockIdx.x+min_node-1;
    iphi=threadIdx.x;
    if (inode>=max_node) return;
    for(int j=0;j<nelement_r[inode];j++){
      ind=eindex_r[j*nnode+inode];
      Er[iphi*nnode+inode]=Er[iphi*nnode+inode]+dpot_turb[iphi*nnode+ind-1]*value_r[j*nnode+inode]; 
    }
    for(int j=0;j<nelement_z[inode];j++){
      ind=eindex_z[j*nnode+inode];
      Ez[iphi*nnode+inode]=Ez[iphi*nnode+inode]+dpot_turb[iphi*nnode+ind-1]*value_z[j*nnode+inode]; 
    }
    if ((grad_psitheta)&&(basis[inode]==0)&&(psi_only)) Ez[iphi*nnode+inode]=0;
    if (nphi>1){
      iphip1=iphi+1;
      iphim1=iphi-1;
      if (iphip1>=nphi) iphip1=iphip1-nphi;
      if (iphim1<0) iphim1=iphim1+nphi;
      Ephi[iphi*nnode+inode]=dpot_turb[iphip1*nnode+inode]-dpot_turb[iphim1*nnode+inode];
    }
  }
  ''','grid_deriv') 
  global Er,Ez,Ephi,nrho
  if (xgc=='xgc1')and(gyro_E):
    from parameters import nrho
  else:
    nrho=0
  Er=np.zeros((nphi,nnode,nsteps,nrho+1),dtype=float)
  Ez=np.zeros((nphi,nnode,nsteps,nrho+1),dtype=float)
  Ephi=np.zeros((nphi,nnode,nsteps,nrho+1),dtype=float)
  nelement_r_gpu=cp.array(nelement_r,dtype=cp.int32)
  nelement_z_gpu=cp.array(nelement_z,dtype=cp.int32)
  eindex_r_gpu=cp.array(eindex_r,dtype=cp.int32).ravel(order='C')
  eindex_z_gpu=cp.array(eindex_z,dtype=cp.int32).ravel(order='C')
  value_r_gpu=cp.array(value_r,dtype=cp.float64).ravel(order='C')
  value_z_gpu=cp.array(value_z,dtype=cp.float64).ravel(order='C')
  basis_gpu=cp.array(basis,dtype=cp.int32)
  for istep in range(nsteps):
    for irho in range(nrho+1):
      if irho==0:
        dpot_turb_gpu=cp.array(dpot_turb[:,:,istep],dtype=cp.float64).ravel(order='C')
      else:
        dpot_turb_gpu=cp.array(dpot_turb_rho[:,:,istep,irho-1],dtype=cp.float64).ravel(order='C')
      Er_gpu=cp.zeros((nphi*nnode,),dtype=cp.float64)
      Ez_gpu=cp.zeros((nphi*nnode,),dtype=cp.float64)
      Ephi_gpu=cp.zeros((nphi*nnode,),dtype=cp.float64)
      grid_deriv_kernel((max_node-min_node+1,),(nphi,),(dpot_turb_gpu,Er_gpu,Ez_gpu,min_node,max_node,nnode,\
            nelement_r_gpu,nelement_z_gpu,eindex_r_gpu,eindex_z_gpu,value_r_gpu,value_z_gpu,\
            grad_psitheta,psi_only,basis_gpu,Ephi_gpu,int(nphi)))
      Er[:,:,istep,irho]=-cp.asnumpy(Er_gpu).reshape((nphi,nnode),order='C')
      Ez[:,:,istep,irho]=-cp.asnumpy(Ez_gpu).reshape((nphi,nnode),order='C')
      if xgc=='xgc1':
        dphi=2*np.pi/float(nphi*nwedge)
        Ephi[:,:,istep,irho]=-cp.asnumpy(Ephi_gpu).reshape((nphi,nnode),order='C')
        for iphi in range(nphi): Ephi[iphi,:,istep,irho]=Ephi[iphi,:,istep,irho]/2/dphi/rz[:,0]

  del Er_gpu,Ez_gpu,Ephi_gpu,nelement_r_gpu,nelement_z_gpu,value_r_gpu,value_z_gpu,basis_gpu
  mempool = cp.get_default_memory_pool()
  pinned_mempool = cp.get_default_pinned_memory_pool()
  mempool.free_all_blocks()
  pinned_mempool.free_all_blocks()

  return

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
  gradphi=np.zeros((Nz,Nr),dtype=float)#assuming axisymmetry
  for i in range(1,Nz-1):
    gradz[i,:]=(fld[i+1,:]-fld[i-1,:])/2/dz
  for j in range(1,Nr-1):
    gradr[:,j]=(fld[:,j+1]-fld[:,j-1])/2/dr

  return gradr,gradz,gradphi

def Curl(r,z,fldr,fldz,fldphi,Nr,Nz):
  #all calculations below assume axisymmetry
  dr=r[2]-r[1]
  dz=z[2]-z[1]
  curlr=np.nan*np.zeros((Nz,Nr),dtype=float)
  curlz=np.nan*np.zeros((Nz,Nr),dtype=float)
  curlphi=np.nan*np.zeros((Nz,Nr),dtype=float)
  for i in range(1,Nz-1):
    curlr[i,:]=-(fldphi[i+1,:]-fldphi[i-1,:])/2/dz
    curlphi[i,:]=(fldr[i+1,:]-fldr[i-1,:])/2/dz
  for j in range(1,Nr-1):
    curlz[:,j]=(r[j+1]*fldphi[:,j+1]-r[j-1]*fldphi[:,j-1])/2/dr/r[j]
    curlphi[:,j]=curlphi[:,j]-(fldz[:,j+1]-fldz[:,j-1])/2/dr
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
  dFdx=np.zeros((nphi,2,nsteps),dtype=float)
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
    
  dFdx[:,0,:]=F[:,0,:]*(z[1]-z[2])+F[:,1,:]*(z[2]-z[0])+F[:,2,:]*(z[0]-z[1])
  dFdx[:,0,:]=dFdx[:,0,:]/xsj
  dFdx[:,1,:]=F[:,0,:]*(r[2]-r[1])+F[:,1,:]*(r[0]-r[2])+F[:,2,:]*(r[1]-r[0])
  dFdx[:,1,:]=dFdx[:,1,:]/xsj
  return dFdx

def node_range(tri_psi):
  from orbit import iorb1,iorb2,nt,steps_orb,R_orb,Z_orb
  global min_node,max_node,itr_save,p_save
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
  return

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
  void node_range(double* R_orb,double* Z_orb,int* steps_orb,int nt,int* itr_save,double* p_save,\
      int ihi,int jhi,int num_tri,double* guess_min,double* inv_guess_d,int* guess_xtable,int* guess_list,\
      int* guess_count,double* mapping,bool tri_psi,double* rlin,double* zlin,double psix,double* psi2d,\
      double* psi_rz,int* nd,int Nr,int Nz,int mynorb,int nblocks_max,int* min_node,int* max_node) 
  {
      double r,z,p[3],tmp; 
      int iorb,it_orb,itr[1],ilo,jlo,ij[2],ix,iy,istart,iend,node; 
      iorb=blockIdx.x;
      it_orb=threadIdx.x;
      while (iorb<mynorb)
      {
        if (it_orb>=steps_orb[iorb]){
          iorb=iorb+nblocks_max;
          continue;
        }
        r=R_orb[iorb*nt+it_orb];
        z=Z_orb[iorb*nt+it_orb];
        search_tr2(r,z,itr,p,ihi,jhi,num_tri,guess_min,inv_guess_d,guess_xtable,guess_list,guess_count,mapping);
        tmp=max(p[0],p[1]);
        tmp=max(tmp,p[2]);
        if ((tri_psi)&&(itr[0]>0)&&(tmp<1.0))
        {
          t_coeff_mod(r,z,rlin,zlin,p,psix,psi2d,psi_rz,itr[0],num_tri,nd,Nr,Nz);
        }
        itr_save[iorb*nt+it_orb]=itr[0];
        p_save[iorb*3*nt+it_orb*3+0]=p[0];
        p_save[iorb*3*nt+it_orb*3+1]=p[1];
        p_save[iorb*3*nt+it_orb*3+2]=p[2]; 
        if (itr[0]>0){
          for(int k=0;k<3;k++){
          node=nd[k*num_tri+itr[0]-1];
          if (node>max_node[iorb*nt+it_orb]) max_node[iorb*nt+it_orb]=node;
          if (node<min_node[iorb*nt+it_orb]) min_node[iorb*nt+it_orb]=node;
          }
        }
      iorb=iorb+nblocks_max;
      }
  }
  ''', 'node_range')
  from orbit import iorb1,iorb2,nt,steps_orb,R_orb,Z_orb
  global min_node,max_node,itr_save,p_save
  nblocks_max=4096
  mynorb=iorb2-iorb1+1
  nblocks=min(nblocks_max,mynorb)
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
  R_orb_gpu=cp.asarray(R_orb,dtype=cp.float64).ravel(order='C')
  Z_orb_gpu=cp.asarray(Z_orb,dtype=cp.float64).ravel(order='C')
  steps_orb_gpu=cp.asarray(steps_orb[iorb1-1:iorb2],dtype=cp.int32)
  itr_save_gpu=cp.zeros((mynorb*nt,),dtype=cp.int32)
  p_save_gpu=cp.zeros((mynorb*nt*3,),dtype=cp.float64)
  min_node_gpu=nnode*cp.ones((mynorb*nt,),dtype=cp.int32)
  max_node_gpu=cp.zeros((mynorb*nt,),dtype=cp.int32)
  node_range_kernel((nblocks,),(nt,),(R_orb_gpu,Z_orb_gpu,steps_orb_gpu,int(nt),itr_save_gpu,p_save_gpu,\
    int(ihi),int(jhi),int(num_tri),guess_min_gpu,inv_guess_d_gpu,guess_xtable_gpu,guess_list_gpu,\
    guess_count_gpu,mapping_gpu,tri_psi,rlin_gpu,zlin_gpu,float(psix),psi2d_gpu,psi_rz_gpu,\
    nd_gpu,int(rlin.size),int(zlin.size),int(mynorb),int(nblocks_max),min_node_gpu,max_node_gpu))
  itr_save=cp.asnumpy(itr_save_gpu).reshape((mynorb,nt),order='C')
  p_save=cp.asnumpy(p_save_gpu).reshape((mynorb,nt,3),order='C')
  min_node=np.asscalar(cp.asnumpy(cp.min(min_node_gpu)))
  max_node=np.asscalar(cp.asnumpy(cp.max(max_node_gpu)))
  del guess_min_gpu,inv_guess_d_gpu,guess_xtable_gpu,guess_list_gpu,guess_count_gpu,mapping_gpu,\
      nd_gpu,rlin_gpu,zlin_gpu,psi2d_gpu,psi_rz_gpu,R_orb_gpu,Z_orb_gpu,steps_orb_gpu,\
      itr_save_gpu,p_save_gpu,min_node_gpu,max_node_gpu
  mempool = cp.get_default_memory_pool()
  pinned_mempool = cp.get_default_pinned_memory_pool()
  mempool.free_all_blocks()
  pinned_mempool.free_all_blocks()
  return

def deriv_node_range(xgc,use_ff):
  global deriv_min_node,deriv_max_node
  deriv_min_node=nnode
  deriv_max_node=0
  for i in range(min_node-1,max_node):
    for j in range(nelement_r[i]):
      ind=eindex_r[j,i]
      if ind>deriv_max_node: deriv_max_node=ind
      if ind<deriv_min_node: deriv_min_node=ind
    for j in range(nelement_z[i]):
      ind=eindex_z[j,i]
      if ind>deriv_max_node: deriv_max_node=ind
      if ind<deriv_min_node: deriv_min_node=ind
    if (xgc=='xgc1')and(use_ff):
      for idir in range(2):
        itr=ff_1dp_tr[i,idir]
        for k in range(3):
          node=nd[k,itr-1]
          if (node>deriv_max_node): deriv_max_node=node
          if (node<deriv_min_node): deriv_min_node=node
  return
