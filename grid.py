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
  from parameters import use_ff
  for istep in range(nsteps):
    gstep=start_gstep+istep*period
    if (idx==1)or(idx==5):
      if xgc=='xgca':
        fname=xgc_dir+'/xgc.orbit.f0.'+'{:0>5d}'.format(gstep)+'.bp'
      elif xgc=='xgc1':
        if idx==1: fname=xgc_dir+'/xgc.orbit.f0.'+'{:0>5d}'.format(gstep)+'.bp'
        if idx==5: fname=xgc_dir+'/xgc.orbit.avgf0.'+'{:0>5d}'.format(gstep)+'.bp'
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
    #read additional data if we need field-line following F_i
    if (idx==1)and(xgc=='xgc1')and(use_ff):
      global df0g_high,df0g_low
      if min_node_ff<min_node:
        n_node=min_node-min_node_ff
        tmp=fid.read('i_f',start=[0,0,min_node_ff-1,0],count=[nphi,nmu,n_node,nvp])
        tmp=np.transpose(tmp)
        if istep==0: df0g_low=np.zeros((nvp,n_node,nmu,nphi,nsteps),dtype=float)
        df0g_low[:,:,:,:,istep]=tmp
      if max_node_ff>max_node:
        n_node=max_node_ff-max_node
        tmp=fid.read('i_f',start=[0,0,max_node,0],count=[nphi,nmu,n_node,nvp])
        tmp=np.transpose(tmp)
        if istep==0: df0g_high=np.zeros((nvp,n_node,nmu,nphi,nsteps),dtype=float)
        df0g_high[:,:,:,:,istep]=tmp
    #end for istep
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
 
def Eturb_gpu(xgc,use_ff,gyro_E,nsteps,grad_psitheta,psi_only):
  import cupy as cp
  grid_deriv_kernel=cp.RawKernel(r'''
  extern "C" __global__
  void grid_deriv(double* dpot_turb,double* Er, double* Ez,int min_node,int max_node,int nnode,\
    int* nelement_r,int* nelement_z,int* eindex_r,int* eindex_z,double* value_r,double* value_z,
    bool grad_psitheta,bool psi_only,int* basis)
  {
    int inode,iphi,ind;
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
  }
  ''','grid_deriv')
  if (xgc=='xgc1')and(use_ff):
    ff_deriv_kernel=cp.RawKernel(r'''
    extern "C" __global__
    void ff_deriv(double* dpot_turb,double* Er,double* Ez,double* Ephi,int min_node,int max_node,int nnode,\
         double* B,int* nd,int* ff_1dp_tr,double* ff_1dp_p,double* ff_1dp_dx,int nphi,int num_tri,int* basis)
    {
      int inode,iphi,iphip1,iphim1,sgn,itr_l,itr_r,node_l,node_r;
      double Epara,Br,Bz,Bphi,Bmag,Bpol,p_l[3],p_r[3],dl_l,dl_r,dl_tot,pot_l,pot_r,pot_m;
      inode=blockIdx.x+min_node-1;
      iphi=threadIdx.x;
      if (inode>=max_node) return;
      iphip1=iphi+1;
      iphim1=iphi-1;
      if (iphip1>=nphi) iphip1=iphip1-nphi;
      if (iphim1<0) iphim1=iphim1+nphi;
      Br=B[inode*3+0];Bz=B[inode*3+1];Bphi=B[inode*3+2];
      Bmag=sqrt(Br*Br+Bz*Bz+Bphi*Bphi);
      Bpol=sqrt(Br*Br+Bz*Bz);
      if (Bphi>0){
        sgn=+1;
      }else{
        sgn=-1;
      }
      itr_l=ff_1dp_tr[inode*2+0];itr_r=ff_1dp_tr[inode*2+1];
      if ((itr_l<0)||(itr_r<0)) return;
      p_l[0]=ff_1dp_p[0*nnode*2+inode*2+0];p_r[0]=ff_1dp_p[0*nnode*2+inode*2+1];
      p_l[1]=ff_1dp_p[1*nnode*2+inode*2+0];p_r[1]=ff_1dp_p[1*nnode*2+inode*2+1];
      p_l[2]=ff_1dp_p[2*nnode*2+inode*2+0];p_r[2]=ff_1dp_p[2*nnode*2+inode*2+1];
      dl_l=ff_1dp_dx[inode*2+0];dl_r=ff_1dp_dx[inode*2+1];
      dl_tot=dl_l+dl_r;
      pot_m=dpot_turb[iphi*nnode+inode];
      pot_l=0.0;pot_r=0.0;
      for (int k=0;k<3;k++){
        node_l=nd[k*num_tri+itr_l-1];
        node_r=nd[k*num_tri+itr_r-1];
        pot_l=pot_l+p_l[k]*dpot_turb[iphim1*nnode+node_l-1];
        pot_r=pot_r+p_r[k]*dpot_turb[iphip1*nnode+node_r-1];
      }
      Epara=sgn*(-dl_r/(dl_l*dl_tot)*pot_l\
                 +(dl_r-dl_l)/(dl_l*dl_r)*pot_m\
                 +dl_l/(dl_r*dl_tot)*pot_r);
      if (basis[inode]==1){
        Ephi[iphi*nnode+inode]=(Epara*Bmag-Er[iphi*nnode+inode]*Br-Ez[iphi*nnode+inode]*Bz)/Bphi;
      }else{
        Ephi[iphi*nnode+inode]=(Epara*Bmag-Ez[iphi*nnode+inode]*Bpol)/Bphi;
      }
    }
    ''','ff_deriv')
  if (xgc=='xgc1')and(not use_ff):
    tor_deriv_kernel=cp.RawKernel(r'''
    extern "C" __global__
    void tor_deriv(double* dpot_turb,double* Ephi,double dphi,int min_node,int max_node,int nnode,\
         double* r,int nphi)
    {
      int inode,iphi,iphip1,iphim1;
      inode=blockIdx.x+min_node-1;
      iphi=threadIdx.x;
      if (inode>=max_node) return;
      iphip1=iphi+1;
      iphim1=iphi-1;
      if (iphip1>=nphi) iphip1=iphip1-nphi;
      if (iphim1<0) iphim1=iphim1+nphi;
      Ephi[iphi*nnode+inode]=(dpot_turb[iphip1*nnode+inode]-dpot_turb[iphim1*nnode+inode])/(2.*dphi*r[inode]);
    }
    ''','tor_deriv')
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
  if (xgc=='xgc1')and(use_ff):
    B_gpu=cp.array(B,dtype=cp.float64).ravel(order='C')
    nd_gpu=cp.array(nd,dtype=cp.int32).ravel(order='C')
    ff_1dp_tr_gpu=cp.array(ff_1dp_tr,dtype=cp.int32).ravel(order='C')
    ff_1dp_p_gpu=cp.array(ff_1dp_p,dtype=cp.float64).ravel(order='C')
    ff_1dp_dx_gpu=cp.array(ff_1dp_dx,dtype=cp.float64).ravel(order='C')
    num_tri=np.shape(nd)[1]
  if (xgc=='xgc1')and(not use_ff):
    dphi=2*np.pi/float(nphi*nwedge)
    r_gpu=cp.array(rz[:,0],dtype=cp.float64)
  for istep in range(nsteps):
    for irho in range(nrho+1):
      if irho==0:
        dpot_turb_gpu=cp.array(dpot_turb[:,:,istep],dtype=cp.float64).ravel(order='C')
      else:
        dpot_turb_gpu=cp.array(dpot_turb_rho[:,:,istep,irho-1],dtype=cp.float64).ravel(order='C')
      Er_gpu=cp.zeros((nphi*nnode,),dtype=cp.float64)
      Ez_gpu=cp.zeros((nphi*nnode,),dtype=cp.float64)
      grid_deriv_kernel((max_node-min_node+1,),(nphi,),(dpot_turb_gpu,Er_gpu,Ez_gpu,min_node,max_node,nnode,\
            nelement_r_gpu,nelement_z_gpu,eindex_r_gpu,eindex_z_gpu,value_r_gpu,value_z_gpu,\
            grad_psitheta,psi_only,basis_gpu))
      Er[:,:,istep,irho]=-cp.asnumpy(Er_gpu).reshape((nphi,nnode),order='C')
      Ez[:,:,istep,irho]=-cp.asnumpy(Ez_gpu).reshape((nphi,nnode),order='C')
      if (xgc=='xgc1')and(use_ff):
        Ephi_gpu=cp.zeros((nphi*nnode,),dtype=cp.float64)
        ff_deriv_kernel((max_node-min_node+1,),(nphi,),(dpot_turb_gpu,Er_gpu,Ez_gpu,Ephi_gpu,min_node,max_node,\
             nnode,B_gpu,nd_gpu,ff_1dp_tr_gpu,ff_1dp_p_gpu,ff_1dp_dx_gpu,int(nphi),int(num_tri),basis_gpu))
        Ephi[:,:,istep,irho]=-cp.asnumpy(Ephi_gpu).reshape((nphi,nnode),order='C')
      if (xgc=='xgc1')and(not use_ff):
        Ephi_gpu=cp.zeros((nphi*nnode,),dtype=cp.float64)
        tor_deriv_kernel((max_node-min_node+1,),(nphi,),(dpot_turb_gpu,Ephi_gpu,float(dphi),min_node,max_node,\
                          nnode,r_gpu,int(nphi)))
        Ephi[:,:,istep,irho]=-cp.asnumpy(Ephi_gpu).reshape((nphi,nnode),order='C')

  del Er_gpu,Ez_gpu,nelement_r_gpu,nelement_z_gpu,value_r_gpu,value_z_gpu,basis_gpu
  if xgc=='xgc1': del Ephi_gpu
  if (xgc=='xgc1')and(use_ff): del B_gpu,nd_gpu,ff_1dp_tr_gpu,ff_1dp_p_gpu,ff_1dp_dx_gpu
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

def gradParF_ff(node,imu,ivp,nsteps,wmu,wvp):
  gradParF=np.zeros((nphi,nsteps))
  if B[node-1,2]>0:
    sgn=+1
  else:
    sgn=-1
  itr_l=ff_1dp_tr[node-1,0]
  itr_r=ff_1dp_tr[node-1,1]
  if (itr_l<0)or(itr_r<0): return
  p_l=ff_1dp_p[:,node-1,0]
  p_r=ff_1dp_p[:,node-1,1]
  dl_l=ff_1dp_dx[node-1,0]
  dl_r=ff_1dp_dx[node-1,1]
  dl_tot=dl_l+dl_r
  F_l=np.zeros((2,2,nphi,nsteps),dtype=float)
  F_r=np.zeros((2,2,nphi,nsteps),dtype=float)
  F_m=np.zeros((2,2,nphi,nsteps),dtype=float)
  F_m[:,:,:,:]=df0g[ivp:ivp+2,node-min_node,imu:imu+2,:,:]
  for k in range(3):
    node_l=nd[k,itr_l-1]
    node_r=nd[k,itr_r-1]
    if node_l<min_node:
      F_l[:,:,:,:]=F_l[:,:,:,:]+p_l[k]*df0g_low[ivp:ivp+2,node_l-min_node_ff,imu:imu+2,:,:]
    elif node_l>max_node:
      F_l[:,:,:,:]=F_l[:,:,:,:]+p_l[k]*df0g_high[ivp:ivp+2,node_l-max_node-1,imu:imu+2,:,:]
    else:
      F_l[:,:,:,:]=F_l[:,:,:,:]+p_l[k]*df0g[ivp:ivp+2,node_l-min_node,imu:imu+2,:,:]

    if node_r<min_node:
      F_r[:,:,:,:]=F_r[:,:,:,:]+p_r[k]*df0g_low[ivp:ivp+2,node_r-min_node_ff,imu:imu+2,:,:]
    elif node_r>max_node:
      F_r[:,:,:,:]=F_r[:,:,:,:]+p_r[k]*df0g_high[ivp:ivp+2,node_r-max_node-1,imu:imu+2,:,:]
    else:
      F_r[:,:,:,:]=F_r[:,:,:,:]+p_r[k]*df0g[ivp:ivp+2,node_r-min_node,imu:imu+2,:,:]

  value_m=F_m[0,0,:,:]*wvp[0]*wmu[0]+F_m[0,1,:,:]*wvp[0]*wmu[1]\
         +F_m[1,0,:,:]*wvp[1]*wmu[0]+F_m[1,1,:,:]*wvp[1]*wmu[1]
  value_l=F_l[0,0,:,:]*wvp[0]*wmu[0]+F_l[0,1,:,:]*wvp[0]*wmu[1]\
         +F_l[1,0,:,:]*wvp[1]*wmu[0]+F_l[1,1,:,:]*wvp[1]*wmu[1]
  value_r=F_r[0,0,:,:]*wvp[0]*wmu[0]+F_r[0,1,:,:]*wvp[0]*wmu[1]\
         +F_r[1,0,:,:]*wvp[1]*wmu[0]+F_r[1,1,:,:]*wvp[1]*wmu[1]

  for iphi in range(nphi):
    iphip1=(iphi+1)%nphi
    iphim1=(iphi-1)%nphi
    gradParF[iphi,:]=sgn*(-dl_r/(dl_l*dl_tot)*value_l[iphim1,:]\
                    +(dl_r-dl_l)/(dl_l*dl_r)*value_m[iphi,:]\
                    +dl_l/(dl_r*dl_tot)*value_r[iphip1,:])
  return gradParF

def gradParF_ff_gpu(B_gpu,df0g_gpu,iphi,nsteps):
  import cupy as cp
  F_ff_deriv_kernel=cp.RawKernel(r'''
  extern "C" __global__
  void F_ff_deriv(double* df0g,double* df0g_r,double* df0g_r_low,double* df0g_r_high,double* df0g_l,\
       double* df0g_l_low,double* df0g_l_high,double* gradParF,int* nd,int* ff_1dp_tr,double* ff_1dp_p,\
       double* ff_1dp_dx,int num_tri,int nnode,int min_node,int min_node_ff,int max_node,int max_node_ff,\
       double* B,int nmu,int nsteps)
  {
    int inode,inode_l,inode_r,ivp,istep,imu,sgn,itr_l,itr_r,node_l,node_r,n_node,n_node_low,n_node_high;
    ivp=threadIdx.x;
    double value_m,value_l,value_r,p_l[3],p_r[3],dl_l,dl_r,dl_tot;
    inode=blockIdx.x+min_node-1;
    n_node=max_node-min_node+1;
    n_node_low=min_node-min_node_ff;
    n_node_high=max_node_ff-max_node;
    if (inode>=max_node) return;
    if (B[inode*3+2]>0.){
      sgn=+1;
    }else{
      sgn=-1;
    }
    itr_l=ff_1dp_tr[inode*2+0];itr_r=ff_1dp_tr[inode*2+1];
    if ((itr_l<0)||(itr_r<0)) return;
    p_l[0]=ff_1dp_p[0*nnode*2+inode*2+0];p_r[0]=ff_1dp_p[0*nnode*2+inode*2+1];
    p_l[1]=ff_1dp_p[1*nnode*2+inode*2+0];p_r[1]=ff_1dp_p[1*nnode*2+inode*2+1];
    p_l[2]=ff_1dp_p[2*nnode*2+inode*2+0];p_r[2]=ff_1dp_p[2*nnode*2+inode*2+1];
    dl_l=ff_1dp_dx[inode*2+0];dl_r=ff_1dp_dx[inode*2+1];
    dl_tot=dl_l+dl_r;
    for (imu=0;imu<nmu;imu++){
      for (istep=0;istep<nsteps;istep++){
        value_m=df0g[ivp*n_node*nmu*nsteps+(inode-min_node+1)*nmu*nsteps+imu*nsteps+istep];
        value_l=0.;
        value_r=0.;
        for (int k=0;k<3;k++){
          node_l=nd[k*num_tri+itr_l-1];
          node_r=nd[k*num_tri+itr_r-1];
          if (node_l<min_node){
            inode_l=node_l-min_node_ff;
            value_l=value_l+p_l[k]*df0g_l_low[ivp*n_node_low*nmu*nsteps+inode_l*nmu*nsteps+imu*nsteps+istep];
          }else if(node_l>max_node){
            inode_l=node_l-max_node-1;
            value_l=value_l+p_l[k]*df0g_l_high[ivp*n_node_high*nmu*nsteps+inode_l*nmu*nsteps+imu*nsteps+istep];
          }else{
            inode_l=node_l-min_node;
            value_l=value_l+p_l[k]*df0g_l[ivp*n_node*nmu*nsteps+inode_l*nmu*nsteps+imu*nsteps+istep];
          }
          if (node_r<min_node){
            inode_r=node_r-min_node_ff;
            value_r=value_r+p_r[k]*df0g_r_low[ivp*n_node_low*nmu*nsteps+inode_r*nmu*nsteps+imu*nsteps+istep];
          }else if(node_r>max_node){
            inode_r=node_r-max_node-1;
            value_r=value_r+p_r[k]*df0g_r_high[ivp*n_node_high*nmu*nsteps+inode_r*nmu*nsteps+imu*nsteps+istep];
          }else{
            inode_r=node_r-min_node;
            value_r=value_r+p_r[k]*df0g_r[ivp*n_node*nmu*nsteps+inode_r*nmu*nsteps+imu*nsteps+istep];
          }
        }//end for k
        gradParF[ivp*n_node*nmu*nsteps+(inode-min_node+1)*nmu*nsteps+imu*nsteps+istep]\
              =sgn*(-dl_r/(dl_l*dl_tot)*value_l\
                    +(dl_r-dl_l)/(dl_l*dl_r)*value_m\
                    +dl_l/(dl_r*dl_tot)*value_r);
      }//end for istep
    }//end for imu
  }
  ''','F_ff_deriv')
  nd_gpu=cp.array(nd,dtype=cp.int32).ravel(order='C')
  ff_1dp_tr_gpu=cp.array(ff_1dp_tr,dtype=cp.int32).ravel(order='C')
  ff_1dp_p_gpu=cp.array(ff_1dp_p,dtype=cp.float64).ravel(order='C')
  ff_1dp_dx_gpu=cp.array(ff_1dp_dx,dtype=cp.float64).ravel(order='C')
  num_tri=np.shape(nd)[1]
  nmu=np.shape(df0g)[2]
  nvp=np.shape(df0g)[0]
  gradParF_gpu=cp.zeros((nvp*(max_node-min_node+1)*nmu*nsteps,),dtype=cp.float64)
  iphip1=(iphi+1)%nphi
  iphim1=(iphi-1)%nphi
  df0g_r_gpu=cp.array(df0g[:,:,:,iphip1,:],dtype=cp.float64).ravel(order='C')
  df0g_l_gpu=cp.array(df0g[:,:,:,iphim1,:],dtype=cp.float64).ravel(order='C')
  if min_node_ff<min_node:
    df0g_r_low_gpu=cp.array(df0g_low[:,:,:,iphip1,:],dtype=cp.float64).ravel(order='C')
    df0g_l_low_gpu=cp.array(df0g_low[:,:,:,iphim1,:],dtype=cp.float64).ravel(order='C')
  else:
    df0g_r_low_gpu=cp.zeros((1,),dtype=cp.float64)
    df0g_l_low_gpu=cp.zeros((1,),dtype=cp.float64)
  if max_node_ff>max_node:
    df0g_r_high_gpu=cp.array(df0g_high[:,:,:,iphip1,:],dtype=cp.float64).ravel(order='C')
    df0g_l_high_gpu=cp.array(df0g_high[:,:,:,iphim1,:],dtype=cp.float64).ravel(order='C')
  else:
    df0g_r_high_gpu=cp.zeros((1,),dtype=cp.float64)
    df0g_l_high_gpu=cp.zeros((1,),dtype=cp.float64)
  F_ff_deriv_kernel((max_node-min_node+1,),(2*f0_nvp+1,),(df0g_gpu,df0g_r_gpu,df0g_r_low_gpu,df0g_r_high_gpu,\
      df0g_l_gpu,df0g_l_low_gpu,df0g_l_high_gpu,gradParF_gpu,nd_gpu,ff_1dp_tr_gpu,ff_1dp_p_gpu,ff_1dp_dx_gpu,\
      int(num_tri),nnode,min_node,min_node_ff,max_node,max_node_ff,B_gpu,nmu,nsteps))
  del nd_gpu,ff_1dp_tr_gpu,ff_1dp_p_gpu,ff_1dp_dx_gpu,df0g_r_gpu,df0g_l_gpu,df0g_r_low_gpu,df0g_l_low_gpu,\
      df0g_r_high_gpu,df0g_l_high_gpu
  return gradParF_gpu

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
  if (xgc=='xgc1')and(use_ff):
    global min_node_ff,max_node_ff
    min_node_ff=nnode
    max_node_ff=0
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
          if (node>max_node_ff): max_node_ff=node
          if (node<min_node_ff): min_node_ff=node
  return
