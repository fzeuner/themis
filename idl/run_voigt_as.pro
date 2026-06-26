;Needed:  gauss_fit, funct_fit and mvoigt from ssw/gen/idl/fitting

;----------------------------------------------------------------------------------------------

function ZBRENT_TC, x1, x2, FUNC_NAME=func_name, _EXTRA = _extra,   $
                         MAX_ITERATIONS=maxit, TOLERANCE=TOL
;+
; NAME:
;     ZBRENT
; PURPOSE:
;     Find the zero of a 1-D function up to specified tolerance.
; EXPLANTION:
;     This routine assumes that the function is known to have a zero.
;     Adapted from procedure of the same name in "Numerical Recipes" by
;     Press et al. (1992), Section 9.3
;
; CALLING:
;       x_zero = ZBRENT( x1, x2, FUNC_NAME="name", MaX_Iter=, Tolerance=, 
;                                 _EXTRA =  )
;
; INPUTS:
;       x1, x2 = scalars, 2 points which bracket location of function zero,
;                                               that is, F(x1) < 0 < F(x2).
;       Note: computations are performed with
;       same precision (single/double) as the inputs and user supplied function.
;
; REQUIRED INPUT KEYWORD:
;       FUNC_NAME = function name (string)
;               Calling mechanism should be:  F = func_name( px )
;               where:  px = scalar independent variable, input.
;                       F = scalar value of function at px,
;                           should be same precision (single/double) as input.
;
; OPTIONAL INPUT KEYWORDS:
;       MAX_ITER = maximum allowed number iterations, default=100.
;       TOLERANCE = desired accuracy of minimum location, default = 1.e-3.
;
;       Any other keywords are passed directly to the user-supplied function
;       via the _EXTRA facility.
; OUTPUTS:
;       Returns the location of zero, with accuracy of specified tolerance.
;
; PROCEDURE:
;       Brent's method to find zero of a function by using bracketing,
;       bisection, and inverse quadratic interpolation,
;
; EXAMPLE:
;       Find the root of the COSINE function between 1. and 2.  radians
;
;        IDL> print, zbrent( 1, 2, FUNC = 'COS')
;
;       and the result will be !PI/2 within the specified tolerance
; MODIFICATION HISTORY:
;       Written, Frank Varosi NASA/GSFC 1992.
;       FV.1994, mod to check for single/double prec. and set zeps accordingly.
;       Use MACHAR() to define machine precision   W. Landsman September 2002
;       Added _EXTRA keyword  W. Landsman  December 2011
;       Need to check whether user function accepts keywords W.L. Jan
;       2012
;       Add missing xc=xb line 86 -  T. Corbard 08/2020
;-
  compile_opt idl2
  if N_params() LT 2 then begin
     print,'Syntax - result = ZBRENT( x1, x2, FUNC_NAME = ,'
     print,'                  [ MAX_ITER = , TOLERANCE = , _EXTRA=])'
     return, -1
  endif
  
  kpresent = keyword_set(_EXTRA)
  if N_elements( TOL ) NE 1 then TOL = 1.e-3
  if N_elements( maxit ) NE 1 then maxit = 100

  if size(x1,/TNAME) EQ 'DOUBLE' OR size(x2,/TNAME) EQ 'DOUBLE' then begin
     xa = double( x1 )
     xb = double( x2 )
     zeps = (machar(/DOUBLE)).eps ;machine epsilon in double.
  endif else begin
     xa = x1
     xb = x2
     zeps = (machar(/DOUBLE)).eps ;machine epsilon, in single 
  endelse
  
  if kpresent then begin 
     fa = call_function( func_name, xa, _EXTRA = _extra )
     fb = call_function( func_name, xb, _EXTRA = _extra )
  endif else begin 
     fa = call_function( func_name, xa )
     fb = call_function( func_name, xb )
  endelse
  xc = xb                       ;This line was missing (cf num. rec.) (TC) 
  fc = fb
  
  if(is_nan(fa*fb)) then begin
     message,"NaN value at input",/INFO
     return,!VALUES.F_NAN
  endif
  if (fb*fa GT 0) then begin
     message,"root must be bracketed by the 2 inputs",/INFO
                                ;return,xa
     return,!VALUES.F_NAN
  endif
  
  for iter = 1,maxit do begin
     
     if (fb*fc GT 0) then begin
        xc = xa
        fc = fa
        Din = xb - xa
        Dold = Din
     endif
     
     if (abs( fc ) LT abs( fb )) then begin
        xa = xb   &   xb = xc   &   xc = xa
        fa = fb   &   fb = fc   &   fc = fa
     endif
     
     TOL1 = 0.5*TOL + 2*abs( xb ) * zeps ;Convergence check
     xm = (xc - xb)/2.
     
     if (abs( xm ) LE TOL1) || (fb EQ 0) then return,xb
     
     if (abs( Dold ) GE TOL1) && (abs( fa ) GT abs( fb )) then begin
        
        S = fb/fa               ;attempt inverse quadratic interpolation
        
        if (xa EQ xc) then begin
           p = 2 * xm * S
           q = 1-S
        endif else begin
           T = fa/fc
           R = fb/fc
           p = S * (2*xm*T*(T-R) - (xb-xa)*(R-1) )
           q = (T-1)*(R-1)*(S-1)
        endelse
        
        if (p GT 0) then q = -q
        p = abs( p )
        test = ( 3*xm*q - abs( q*TOL1 ) ) < abs( Dold*q )
        
        if (2*p LT test)  then begin
           Dold = Din           ;accept interpolation
           Din = p/q
        endif else begin
           Din = xm             ;use bisection instead
           Dold = xm
        endelse
        
     endif else begin

        Din = xm                ;Bounds decreasing to slowly, use bisection
        Dold = xm
     endelse
     
     xa = xb
     fa = fb                    ;evaluate new trial root.
     
     if (abs( Din ) GT TOL1) then xb = xb + Din $
     else xb = xb + TOL1 * (1-2*(xm LT 0))
     
     if kpresent then $
        fb = call_function( func_name, xb, _EXTRA = _extra ) else $
           fb = call_function( func_name, xb )
     if(is_nan(fb)) then begin
        message,"NaN value during search",/INFO
        return,!VALUES.F_NAN
     endif
  endfor

  message,"exceeded maximum number of iterations: "+strtrim(iter,2),/INFO

  ;return, xb
  return,!VALUES.F_NAN
end
;----------------------------------------------------------------------------------------------
;----------------------------------------------------------------------------------------------
pro tc_str_replace,vst,ch_old,ch_new
; Replace string ch_old by string ch_new in a string vector v
N=N_elements(vst)

for i=0,N-1 do begin
    st=vst[i]
    pold=-1
    ok=1
    while(strmatch(st,'*'+ch_old+'*') and ok) do begin
        
        len_old=strlen(ch_old)
        len=strlen(st)
        
        p=strpos(st,ch_old)
        
        ok=(p gt pold)
        pold=p
        if(ok) then st=strmid(st,0,p)+ch_new+strmid(st,p+len_old,len-p-len_old)
        
    endwhile
    vst[i]=st
endfor

end
;--------------------------------------------------------------------------------------------
;--------------------------------------------------------------------------------------------
function tc_intersec,set,subset
; return what is in set but not in subset
; set and subset are lists
filtered=set[*]

foreach set_elt,set,k do $
   if(subset.where(set_elt) ne !NULL) then filtered.remove,filtered.where(set_elt)

return,filtered

end
;-------------------------------------------------------------------------------------------
;------------------------------------------------------------------------------------------
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
;----------------------------------------------------------------------------------------------
;----------------------------------------------------------------------------------------------
function get_contrast,im,ywin,width
;get the median value of the ratio between 
;running sigma and running mean
  y0=ywin[0]
  y1=ywin[1]
  v=total(im[*,y0:y1],1)
  run_mean_sigma,v,width,xm,sm
  return,median(stddev(v-xm)/xm)
end
;----------------------------------------------------------------------------------------------
;----------------------------------------------------------------------------------------------
;+
; Project:  YOHKOH-BCS
;
; Name:     VOIGT_FIT
;
; Purpose:  Single voigt function fit to line profile
;
; Syntax:   v=voigt_fit(x,y,a,sigmaa,damp=damp)
;
; Category: Fitting
;
; Inputs:
;       y = data to fit
;       x = bin or wavelength
;
; Outputs:
;       background=a(0)+a(1)*x+a(2)*x^2
;       a(3) = total intensity (1)
;       a(4) = center (1)
;       a(5) = doppler width (1)
;       a(6) = damping width (1)
;
; Opt. Outputs:
;      sigmaa = sigma errors
;
; Keywords:
;       damp   = damping width
;       fixp   = vector of parameters to keep fixed 
;                (e.g. fixp=[0,1,3] to fix parameters 1,2 and 4)
;       last   = use latest parameter values as new first guesses.
;       weights = data weights
;       nfree  = number of free parameters
;       chi2   = chi^2
;       corr   = link matrix
;
; Common: None.
;               
; Restrictions: None.
;               
; Side effects: None.
;               
; History:      Version 1,  17-July-1993,  D M Zarro.  Written
;
; Contact:      DZARRO@SOLAR.STANFORD.EDU
;-            
function voigt_fit_tc,x,y,a,sigmaa,damp=damp,fixp=fixp,weights=weights,$
           chi2=chi2,last=last,nfree=nfree

 if not keyword_set(last) or (n_elements(a) eq 0) then begin
    fit_par=fltarr(6)
    fit_par[0]=max(y)*0.95
    fit_par[1]=0.
    fit_par[2]=0.
    fit_par[3]=min(y)-max(y)*0.95
    fit_par[5]=8
  g=gauss_fit(reform(x),reform(y),fit_par,fixp=[1,2,4],weights=weights,/last)
  ;print,fit_par
  ;cgplot,reform(x),reform(y),ps=1
  ;cgoplot,reform(x),g,color='green'
  ;swait,0
  if n_elements(damp) eq 0 then damp=.01*fit_par(5)
  a=[fit_par,damp]
  a(3)=a(3)*a(5)*sqrt(!pi)
 endif


 v=funct_fit(reform(x),reform(y),weights=weights,a,sigmaa,funct='mvoigt',$
         fixp=fixp,chi2=chi2,nfree=nfree,status=status)
 ;print,'status=',status

 return,v
end
;------------------------------------------------------------------------------------------------
;------------------------------------------------------------------------------------------------
pro find_min,profile,xmin,pmin,Ni=Ni
;cubic interpolation over Ni points and get the position (xmin) and
;value (pmin) of the profile minimum   
  SetDefaultValue,Ni,1000
  N=N_elements(profile)
  x=repar(0,N-1,Ni)
  y=interpolate(profile,x,cubic=-0.5)
  pmin=min(y,imin)
  xmin=x[imin]
end
;------------------------------------------------------------------------------------------------
;------------------------------------------------------------------------------------------------
function func,x,a=a,val=val
  ;0 for x such that mvoigt(x,a)=val
  return,mvoigt(x,a)-val
end

function inv_voigt,y,a,ma
  N=N_elements(y)
  Imin=mvoigt(0.,a)
  x=fltarr(N)
  for i=0,N-1 do begin
     if(y[i] le Imin) then begin
        x[i]=0.
        ;print,'x=0'
     endif else begin
        map=ma
        while(y[i] ge mvoigt(map,a) and map lt 2*ma) do map*=1.1
        x[i]=zbrent_tc(0,map,func_name='func',a=a,val=y[i])
        if(is_nan(x[i])) then begin
           print,'func'
           print,func(0,a=a,val=0.)
           print,func(map,a=a,val=0.)
           print,y[i]
        endif
     endelse
  endfor
  if (N eq 1) then return, x[0] else return,x
end
;------------------------------------------------------------------------------------------------
;-----------------------------------------------------------------------------------------------

function func2,val,ar=ar,ab=ab,mar=mar,mab=mab,dx=dx
  ;0 for val such that dxb(val)+dxr(val)=dx
  return, (inv_voigt(val,ab,mab)+inv_voigt(val,ar,mar))-dx
end

function niv,dx,Imin,Imax,ar,ab,mar,mab
  N=N_elements(dx)
  val=fltarr(N)
  for i=0,N-1 do begin
     val[i]=zbrent_tc(Imin,Imax,func_name='func2',$
                      ar=ar,ab=ab,mar=mar,mab=mab,dx=dx[i])
     if(is_nan(val[i])) then begin
        print,'func2'
        print,func2(Imin,ar=ar,ab=ab,mar=mar,mab=mab,dx=0.)
        print,func2(Imax,ar=ar,ab=ab,mar=mar,mab=mab,dx=0.)
        print,dx[i]
     endif
  endfor
  if(N eq 1) then return,val[0] else return,val
end
;------------------------------------------------------------------------------------------------
;------------------------------------------------------------------------------------------------
pro voigt_as,profile,Nniv,frac_Ic,$
             ab_t,ar_t,$
             chi2b,chi2r,$
             delta0,$
             Ic,DeltaLcb,DeltaLcr,val,$
             quadratic=quadratic,show=show,NIc=NIc

  SetDefaultValue,NIc,1

  if(NIc ne 1 and NIc ne 2) then message,'wrong value for NIc'

  ;Get profile minimum value and location
  find_min,profile,xmin,pmin
  N=N_elements(profile)

  ;distances from line core position (frequency bin unit)
  x=findgen(N)-xmin
  pos=where(x ge 0,complement=neg,npos,ncomplement=nneg)

  ;Fiting blue wing
  xb=[x[neg],0,-reverse(x[neg])]
  vb=[profile[neg],pmin,reverse(profile[neg])]
 
  fixp=keyword_set(quadratic)?[1,4]:[1,2,4]
  vgb=voigt_fit_tc(xb,vb,ab,sigmaa,chi2=chi21,fixp=fixp,$
                   ;weights=1./(abs(xb)+1.d-2))
                   weights=1+1./(abs(xb)+3.d-1))
  ab_t=ab[cgSetDifference(indgen(7),fixp)] ;all but fixed
  chi2b=total((vb-vgb)^2)
 
  ;Fiting red wing
  xr=[reverse(-x[pos]),0,x[pos]]
  vr=[reverse(profile[pos]),pmin,profile[pos]]
 
  vgr=voigt_fit_tc(xr,vr,ar,sigmaa,chi2=chi22,fixp=fixp,$
                   ;weights=1./(abs(xr)+1.d-2))
                  weights=1+1./(abs(xr)+3.d-1))
  ar_t=ar[cgSetDifference(indgen(7),fixp)] ;all but fixed
  chi2r=total((vr-vgr)^2)
 
  ;Discontinuity get at line core
  delta0=vgr[-npos-1]-vgb[nneg]


  ;Get fitted profile values each side 
  ;up to a distance from line core which corresponds to the maximum
  ;distance available in the data (either in blue or red wings)
  ;xx=repar(0,max([N_elements(neg),N_elements(pos)]),100)
  ;vr=mvoigt(xx,ar)
  ;vb=mvoigt(xx,ab)

  ;Get the intensity max and min both sides and their location
  ;Imaxr=max(vr,imar,min=Iminr)
  ;mar=xx[imar]
  ;Imaxb=max(vb,imab,min=Iminb)
  ;mab=xx[imab]
  
  ;Get the intensity at the line core and in the fitted continuum 
  mar=max([N_elements(neg),N_elements(pos)])*2
  mab=mar
  Iminr=mvoigt(0.,ar)
  Iminb=mvoigt(0.,ab)
  Imaxr=mvoigt(mar,ar)
  Imaxb=mvoigt(mab,ab)

  ;Take the intensity range common for both sides
  Imin=min([Iminr,Iminb])
  Imax=min([Imaxr,Imaxb])
  ;dmin=func2(Imin,ar=ar,ab=ab,mar=mar,mab=mab,dx=0.)
  ;dmax=func2(Imax,ar=ar,ab=ab,mar=mar,mab=mab,dx=0.)

  ;Continuum level 
;;  NIc=2 & Ic=fltarr(NIc)
;;  Ic[0]=frac_Ic  ;we assume continuum level at 1.
  ;print,'ab[0]=',ab[0]
;;  Ic[1]=frac_Ic*min([ab[0],1.])
  ;Ic[1]=frac_Ic*ab[0] ;we assume that the background level 
                     ;fitted on the blue wing gives the continuum level  

                                ;27/10/2022 pour les données
                                ;THEMIS on se base sur l'aile
                                ;bleue seulement 
  ;NIc=1 & 

  Ic=fltarr(NIc)

  if(NIc eq 1) then begin
     Ic[0]=frac_Ic*min([ab[0],1.])
  endif else begin
     Ic[0]=frac_Ic  ;we assume continuum level at 1.
     Ic[1]=frac_Ic*ab[0] ;we assume that the background level 
                                ;fitted on the blue wing gives the
                                ;continuum level 
  endelse


  if(keyword_set(show)) then begin
     !P.multi=[0,NIc,1]
     max_ab=fix(max(abs(x))/10.)*10+10
     titles=(NIc eq 2)?['Ic org','Ic fitted']:['Ic fitted']
     for k=0,NIc-1 do begin   
        cgplot,x*13.508,profile/ab[0],ps=1,yr=[0.2,Ic[k]/frac_Ic/ab[0]],xr=[-max_ab,max_ab]*13.508,$
               ys=1,title=titles[k],ytitle=tex2idl('$I/Ic_b$'),xtitle='x [mA]'
        cgoplot,[0,x[pos]]*13.508,vgr[-npos-1:-1]/ab[0],color='red'
        cgoplot,[x[neg],0]*13.508,vgb[0:nneg]/ab[0],color='blue'
        DeltaLcb=inv_voigt(Ic[k],ab,mab)
        DeltaLcr=inv_voigt(Ic[k],ar,mar)
        cgoplot,[-DeltaLcb,DeltaLcr]*13.508,[Ic[k],Ic[k]]/ab[0]
     endfor
     swait,0
  endif


  ;Width at reference levels
  ;print,'DeltaLcb'
  ;print,Ic,ab[0]
  DeltaLcb=inv_voigt(Ic,ab,mab)
  ;print,DeltaLcb
  ;swait,0
  ;print,'DeltaLcr'
  DeltaLcr=inv_voigt(Ic,ar,mar)
  ;print,DeltaLcr
  ;swait,0
  DeltaLc=DeltaLcb+DeltaLcr

  dx = fltarr(Nniv,NIc)
  val= fltarr(Nniv,NIc)
  for k=0,NIc-1 do begin
     ;Take Nniv widths between 0 and the width at reference level
     if(is_nan(DeltaLc[k])) then begin
        val[0,k]=pmin
        val[1:-1,k]=!VALUES.F_NAN
     endif else begin
        dx[*,k]=repar(0,DeltaLc[k],Nniv)
                                ;if(is_nan(DeltaLc[k])) then begin
                                ;   print,'max is nan'
                                ;   print,mar,mab
                                ;   print,ar[0]
                                ;   print,dx[*,k]
                                ;endif
                                
       ;get the intensity for each width by taking the inverse profile function
       ;(intensity at width=0 is given by pmin)
        val[*,k]=[pmin,niv(dx[1:-1,k],Imin,Imax,ar,ab,mar,mab)]
                                ;stop
     endelse
  endfor
end
;---------------------------------------------------------------------------------------------------------
;---------------------------------------------------------------------------------------------------------
pro update_head,h,str_pos
  x=sxpar(h,'DIST_EW')
  y=sxpar(h,'DIST_NS')
  r=sxpar(h,'SOLAR_R')
  dobs=sxpar(h,'DATE-OBS')
  
  offset_EW=0.
  offset_NS=0.
  day=strmid(dobs,8,2)
  ;if (day eq '19') then begin   ;Seq 10 du 19/07/2022
  ;   offset_EW=-110.
  ;   offset_NS=10.
  ;endif
  ;if (day eq '20' or day eq '21' or day eq '22') then begin
  ;   offset_EW=-85.5
  ;   offset_NS=4.7
  ;endif
  x-=offset_EW
  y-=offset_NS
  sintta=sqrt(x^2+y^2)/r
  mu=sqrt(1-(x^2+y^2)/r^2)
  if(x gt 0 and abs(x) gt abs(y)) then dir='W'
  if(x lt 0 and abs(x) gt abs(y)) then dir='E'
  if(y lt 0 and abs(x) lt abs(y)) then dir='S'
  if(y gt 0 and abs(x) lt abs(y)) then dir='N'
  if(mu gt 0.98) then dir='C'
  sxaddpar,h,'COSTHETA',mu
  sxaddpar,h,'SINTHETA',sintta
  sxaddpar,h,'POSITION',dir
  sxaddpar,h,'OFFST_EW',offset_EW
  sxaddpar,h,'OFFST_NS',offset_NS
  str_pos=string(mu*100,format='(i3.3)')+dir
end
;-----------------------------------------------------------------------------------------------------------
;-----------------------------------------------------------------------------------------------------------

pro run_voigt_as,file,line_name, lbda_width, show=show,limit_cs=limit_cs,width_cs=width_cs,$
                 lbda_smooth_width=lbda_smooth_width,$
                 Nniv=Nniv,frac_Ic=frac_Ic,NIc=NIc,quadratic=quadratic
;file : fits cube file name (wavelength,slit dir,scan dir)
;line name: string e.g. '6301'
;lbda_width: first and last array indices for the line location eg. [170,233]
;/show : display the line fit
;limit_cs : contrast limit below which scans are ignored
;width_cs : size (in pixels along the slit) for contast calculation runing window  
;lbda_smooth_width: smoothing window size (in pixels along the wavelength axis) 
;                    
;Nniv : Number of output levels (line cuts) between the line core and frac_Ic*continuum level 
;frac_Ic fraction of the continuum for the highest line cord
;NIc : 1 or 2 for one or two definition of the continuum level
;/quadratic if set a quadratic trend is fitted for the continuum level
;           (otherwise linear trend only)

  
  
  SetDefaultValue,lbda_smooth_width,10
  SetDefaultValue,Nniv,25
  SetDefaultValue,frac_Ic,0.95
  SetDefaultValue,limit_cs,0.02
  SetDefaultValue,width_cs,21
  SetDefaultValue,quadratic,0,/boolean
  SetDefaultValue,NIc,1

 
  cube=float(readfits(file,h,/noscale))

  Nz=sxpar(h,'NAXIS3')
  ;y0=sxpar(h,'Y0WIN')-1
  ;y1=sxpar(h,'Y1WIN')-1
  y0=0
  y1=sxpar(h,'NAXIS2')-1
  ywin_cs=[y0,y1]
  ;line_name=strtrim(sxpar(h,'WAVELNTH'),2)
  ;CASE line_name OF
  ;   '6162': BEGIN
  ;      lbda_win=[240,340]
  ;      lbda_min=lbda_win[0] 
  ;      lbda_max=lbda_win[1]
  ;   END
  ;   '6302': BEGIN
  ;      lbda_win=[180,250]
  ;      lbda_min=lbda_win[0] 
  ;      lbda_max=lbda_win[1]
  ;   END
  ;ELSE:   stop,'unknown lbda_win'
  ;ENDCASE



  cs=fltarr(Nz)
  for k=0,Nz-1 do cs[k]=get_contrast(cube[*,*,k],ywin_cs,width_cs)
  goodcs=where(cs gt limit_cs,ngoodcs)
  if(ngoodcs eq 0) then begin
     message,'No good contrast found',/info
     return  
  endif
  maxmap=max(smooth(cube,[lbda_smooth_width,1,1]),dim=1)
 
  Ny=y1-y0+1
  Mval0=fltarr(Ny,ngoodcs,Nniv)
  if(NIc eq 2) then Mval1=fltarr(Ny,ngoodcs,Nniv)
  Ndata=(keyword_set(quadratic))?19:17
  if(NIc eq 1) then Ndata-=3
  data=fltarr(Ny,ngoodcs,Ndata)
  kg=-1
  good=goodcs
  ngood=ngoodcs
  for k=0,ngoodcs-1 do begin
     kg=kg+1
     print,k,'/',ngoodcs,ngood
     good[kg]=goodcs[k]
     for i=y0,y1 do begin
        ;print,k,i
        p=reform(cube[lbda_min:lbda_max,i,goodcs[k]])
        p/=maxmap[i,goodcs[k]]
        voigt_as,p,Nniv,frac_Ic,ab,ar,chi2b,chi2r,delta0,$
                 Ic,DeltaLcb,DeltaLcr,val,quadratic=quadratic,show=keyword_set(show),NIc=NIc
        bad=where(is_nan(val),nbad)
        if(nbad ne 0) then begin
           kg=kg-1
           ngood=ngood-1
           break ;We skip the whole k plane
        endif
        val*=maxmap[i,goodcs[k]]
        Mval0[i-y0,kg,*]=val[*,0]
        if(NIc eq 2) then Mval1[i-y0,kg,*]=val[*,1]
        data[i-y0,kg,*]=[ab,ar,chi2b,chi2r,delta0,Ic,DeltaLcb,DeltaLcr]
     endfor
  endfor
  

  sxaddpar,h,'NAXIS1',Ny
  sxaddpar,h,'NAXIS2',ngood
  sxaddpar,h,'NAXIS3',Nniv
  sxaddpar,h,'BITPIX',-32
  sxaddpar,h,'BSCALE',1
  sxaddpar,h,'BZERO',0
  sxaddpar,h,'CLEVEL',frac_Ic
  sxaddpar,h,'LIMIT_CS',limit_cs ;13/03/2023
  sxaddpar,h,'NGOOD_CS',ngoodcs,'nb of good contrast spec. images' ;20/07/2024
  sxaddpar,h,'Nz',Nz,'initial number of spectral images'

  ;sxaddpar,h,'CMAX',cont_max

  update_head,h,str_pos
   
  ;lin4  exclusion des plans a problème et ajout d'une extension 
  ;pour identifier les plans retenus
                                ;limit_cs,0.025 (au lieu de 0.02 pour
                                ;lin3) (pas tjrs-> introduce LIMIT_CS key) 
  ext=keyword_set(quadratic)?'_niv_quad':'_niv_lin4'
  ext+='_'+line_name
  outfile=file
  tc_str_replace,outfile,'_tc3',ext+'_'+str_pos

  writefits,outfile,Mval0[*,0:ngood-1,*],h
  if(NIc eq 2) then writefits,outfile,Mval1[*,0:ngood-1,*],/APPEND
  writefits,outfile,data[*,0:ngood-1,*],/APPEND
  ;19/07/2024  ajout de [0:ngood-1]
  writefits,outfile,good[0:ngood-1],/APPEND
  writefits,outfile,cs[good[0:ngood-1]],/APPEND ;13/03/2023
  if(ngood ne Nz) then begin
     scan_numbers=list(good[0:ngood-1],/extract)
     missing_scan=tc_intersec(list(indgen(Nz),/extract),scan_numbers)
     writefits,outfile,missing_scan,/APPEND
  endif
  
end
