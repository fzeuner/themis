
function get_contrast,im,ywin,width
  ;get the median value of the ratio between
  ;running sigma and running mean
  y0=ywin[0]
  y1=ywin[1]
  v=total(im[*,y0:y1],1)
  run_mean_sigma,v,width,xm,sm
  return,median(stddev(v-xm)/xm)
end

pro run_mean_sigma,x,width,xm,sm,step=step

  ;Take a centered running mean over 'width' points.
  ;width must be odd
  ;Edges (i.e. width/2 points each side [0,..width/2-1]   [Nx-width/2,..Nx-1])
  ; are duplicated each side

  ;defined if all points of the window are taken (default) or one over two or...
  if(keyword_set(step)) then st=step else st=1

  ;width must be odd
  if ((width mod 2) eq 0) then stop, 'run_mean_true: width must be odd'
  w2=width/2

  ;size of input vector
  Nx=N_elements(x)

  ;build the increased vector by niroing both edges
  y=fltarr(Nx+2*w2)
  y[0:w2-1]=reverse(x[0:w2-1])
  y[w2:Nx-1+w2]=x[0:Nx-1]
  y[Nx+w2:Nx+2*w2-1]=reverse(x[Nx-w2:Nx-1])

  ;Apply the runing mean
  xm=fltarr(Nx)
  sm=xm
  for i=0,Nx-1 do begin
    xm[i]=mean(y[i:i+width-1:st],/NAN)
    sm[i]=stddev(y[i:i+width-1:st],/NAN)
  endfor

end

pro test_input

  SetDefaultValue,lbda_smooth_width,10
  SetDefaultValue,Nniv,25
  SetDefaultValue,frac_Ic,0.95
  SetDefaultValue,limit_cs,0.02
  SetDefaultValue,width_cs,21
  SetDefaultValue,quadratic,0,/boolean
  SetDefaultValue,NIc,1


file='/home/franziskaz/projects/themis/data/pdata/2025-07-05/spectra/ti/disk_center/disk_center_fit_line.fits

cube=float(readfits(file,h,/noscale))

Nz=sxpar(h,'NAXIS3')

y0=0
y1=sxpar(h,'NAXIS2')-1
ywin_cs=[y0,y1]

cs=fltarr(Nz)
for k=0,Nz-1 do cs[k]=get_contrast(cube[*,*,k],ywin_cs,width_cs)
goodcs=where(cs gt limit_cs,ngoodcs)
if(ngoodcs eq 0) then begin
  message,'No good contrast found',/info
  return
endif
maxmap=max(smooth(cube,[lbda_smooth_width,1,1]),dim=1)
stop
end