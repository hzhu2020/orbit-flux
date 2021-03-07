import adios2 as ad
import numpy as np

def read(idx,fname,min_node,max_node):
  global df0g
  fid=ad.open(fname,'r')
  nnode=fid.read('nnode')
  nmu=fid.read('mudata')
  nvp=fid.read('vpdata')
  nnode=max_node-min_node+1
  df0g=fid.read('i_df0g',start=[0,min_node-1,0],count=[nmu,nnode,nvp])
  df0g=np.transpose(df0g)#[vp,node,mu] order as in Fortran XGC
