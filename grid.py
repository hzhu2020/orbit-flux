import adios2 as ad
import numpy as np
from math import floor
from scipy.interpolate import griddata

def read(xgc_dir,Nr,Nz):
  global rz,guess_table,guess_xtable,guess_count,guess_list,mapping,\
         guess_min,inv_guess_d,nnode,nd,psix,psi_rz,psi2d,R,Z,\
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

def grid_deriv_init(xgc_dir):
  global basis,nelement_r,eindex_r,value_r,nelement_z,eindex_z,value_z

  fname=xgc_dir+'/xgc.grad_rz.bp'
  fid=ad.open(fname,'r')
  basis=fid.read('basis')
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
  
def additional_Bfield(xgc_dir): 
  global nb_curl_nb
  fname=xgc_dir+'/xgc.f0.mesh.bp'
  fid=ad.open(fname,'r')
  nb_curl_nb=fid.read('nb_curl_nb')
  fid.close()
  
def read_dpot_orb(orbit_dir):
  global dpot_orb
  dpot_orb=np.zeros((nnode))
  fname=orbit_dir+'/pot0m.txt'
  fid=open(fname,'r')
  dum=int(fid.readline(8))
  if dum!=nnode:
    print('Wrong nnode for pot0m.txt. Set dpot_orb=0.')
    return
  fid.readline(8)
  for i in range(nnode):
    value=fid.readline(19)
    dpot_orb[i]=float(value)
    fid.readline(1)
  dum=int(fid.readline(8))
  if dum!=-1:
    print('Wrong file end for pot0m.txt. Set dpot_orb=0.')
    dpot_orb[:]=0.0
    return

  fid.close()
  return

def Eturb(xgc_dir,gstep,grad_psitheta,psi_only): 
  fname=xgc_dir+'/xgc.2d.'+'{:0>5d}'.format(gstep)+'.bp'
  fid=ad.open(fname,'r')
  dpot=fid.read('dpot')
  dpot=dpot-dpot_orb
  Er=np.zeros((nnode,),dtype=float)
  Ez=np.zeros((nnode,),dtype=float)
  for i in range(nnode):
    for j in range(nelement_r[i]):
      ind=eindex_r[j,i]
      Er[i]=Er[i]+dpot[ind-1]*value_r[j,i]
    for j in range(nelement_z[i]):
      ind=eindex_z[j,i]
      Ez[i]=Ez[i]+dpot[ind-1]*value_z[j,i]
  if grad_psitheta:
    for i in range(nnode):
      if (basis[i]==0)and(psi_only): Ez[i]==0
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
  elif abs(psi[1]-psi[2])<-eps1:
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
