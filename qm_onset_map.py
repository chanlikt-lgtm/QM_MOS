import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq, anderson
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv, os, warnings
warnings.filterwarnings('ignore')

q=1.602176634e-19; hbar=1.054571817e-34; kB=1.380649e-23
m0=9.1093837015e-31; eps0=8.8541878128e-12
T=300.; VT=kB*T/q; eps_si=11.7*eps0; eps_ox=3.9*eps0; Eg=1.122
ml=0.9163*m0; mt=0.2245*m0
valleys=[('L',2,ml,mt),('T',4,mt,np.sqrt(ml*mt))]
md3=(ml*mt*mt)**(1/3); Nc=2*6*(md3*kB*T/(2*np.pi*hbar**2))**1.5
Nv=1.04e19*1e6; EA_ION=0.045; GA=4.; Vfb=-0.95
L=30e-9; N=101; x=np.linspace(0,L,N); dx=x[1]-x[0]; Ni=N-2
Vg_map=4.0; NSTATES=6
_table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fd_half_table.npz')
if not os.path.exists(_table_path):
    _table_path = 'fd_half_table.npz'
_fd = np.load(_table_path)
FD_ETA = _fd['eta']
FD_F = _fd['F']
FD_D = _fd['dF']

def fdhalf(eta):
    eta=np.asarray(eta); out=np.empty_like(eta,dtype=float)
    lo=eta<FD_ETA[0]; hi=eta>FD_ETA[-1]; mid=~(lo|hi)
    out[lo]=np.exp(eta[lo])
    z=np.maximum(eta[hi],1e-12); out[hi]=4/(3*np.sqrt(np.pi))*z**1.5*(1+np.pi**2/(8*z**2))
    out[mid]=np.interp(eta[mid],FD_ETA,FD_F)
    return out

def dfdhalf(eta):
    eta=np.asarray(eta); out=np.empty_like(eta,dtype=float)
    lo=eta<FD_ETA[0]; hi=eta>FD_ETA[-1]; mid=~(lo|hi)
    out[lo]=np.exp(eta[lo])
    z=np.maximum(eta[hi],1e-12); A=4/(3*np.sqrt(np.pi)); out[hi]=A*(1.5*np.sqrt(z)-0.5*(np.pi**2/8)*z**(-1.5))
    out[mid]=np.interp(eta[mid],FD_ETA,FD_D)
    return out

def log1pexp_scalar(a):
    if a>40: return a
    if a<-40: return np.exp(a)
    return np.log1p(np.exp(a))

def setup_bulk(NAcm):
    NA=NAcm*1e6
    def f(EfEv):
        p=Nv*np.exp(-EfEv/VT); n=Nc*np.exp(-(Eg-EfEv)/VT)
        na=NA/(1+GA*np.exp((EA_ION-EfEv)/VT))
        return p-n-na
    EfEv=brentq(f,0.001,0.7)
    EcEf=Eg-EfEv
    p0=Nv*np.exp(-EfEv/VT); n0=Nc*np.exp(-EcEf/VT)
    return NA,EfEv,EcEf,p0,n0

def model_functions(NAcm,toxnm):
    NA,EfEv,EcEf,p0,n0=setup_bulk(NAcm); tox=toxnm*1e-9
    alpha=eps_ox/(eps_si*tox)
    def holes(phi): return p0*np.exp(np.clip(-phi/VT,-100,100))
    def ion(phi):
        arg=np.clip((EA_ION-(EfEv+phi))/VT,-100,100)
        return NA/(1+GA*np.exp(arg))
    def celec(phi):
        eta=(phi-EcEf)/VT
        return Nc*fdhalf(eta)
    def dcelec(phi):
        eta=(phi-EcEf)/VT
        return Nc*dfdhalf(eta)/VT
    def qdens(phi):
        U=-phi; nt=np.zeros_like(phi)
        for name,g,mz,md in valleys:
            t=(hbar*hbar/(2*mz*dx*dx))/q
            vals,vecs=eigh_tridiagonal(U[1:-1]+2*t,-t*np.ones(Ni-1),select='i',select_range=(0,NSTATES-1))
            for E,v in zip(vals,vecs.T):
                psi=np.zeros(N); psi[1:-1]=v; psi/=np.sqrt(np.trapezoid(psi*psi,x))
                eta=((-EcEf)-E)/VT
                Ns=g*md*kB*T/(np.pi*hbar*hbar)*log1pexp_scalar(eta)
                nt += Ns*psi*psi
        return nt+n0
    return NA,EfEv,EcEf,p0,n0,alpha,holes,ion,celec,dcelec,qdens

def solve_poisson_fixed_n(nfix,guess,alpha,Vgb,holes,ion,tol=2e-6,maxit=50):
    M=N-1; ph=guess[:M].copy()
    for _ in range(maxit):
        full=np.zeros(N); full[:M]=ph; pp=holes(full)[:M]; naa=ion(full)[:M]
        dpp=-pp/VT; f=naa/(np.max(naa)*0+1) # dummy replaced next line
        # derivative N_A f(1-f)/VT, infer NA from saturation robustly
        # use analytic ratio via local occupancy; NA reconstructed as max ceiling not safe, pass from ion(large phi)
        NAceil=float(ion(np.array([5.0]))[0]); occ=np.clip(naa/NAceil,0,1); dna=NAceil*occ*(1-occ)/VT
        R=np.zeros(M); J=lil_matrix((M,M))
        R[0]=(-3*ph[0]+4*ph[1]-ph[2])/(2*dx)-alpha*(ph[0]-Vgb)
        J[0,0]=-3/(2*dx)-alpha; J[0,1]=4/(2*dx); J[0,2]=-1/(2*dx)
        for i in range(1,M):
            pip1=ph[i+1] if i+1<M else 0.0
            R[i]=(ph[i-1]-2*ph[i]+pip1)/dx**2-q/eps_si*(nfix[i]+naa[i]-pp[i])
            J[i,i-1]=1/dx**2; J[i,i]=-2/dx**2-q/eps_si*(dna[i]-dpp[i])
            if i+1<M: J[i,i+1]=1/dx**2
        delta=spsolve(J.tocsr(),-R); md=np.max(np.abs(delta))
        if md>0.15: delta*=0.15/md
        ph += 0.45*delta
        if np.max(np.abs(0.45*delta))<tol: break
    out=np.zeros(N); out[:M]=ph; return out

def solve_classical_gate(NAcm,toxnm,Vg=Vg_map,guess=None):
    NA,EfEv,EcEf,p0,n0,alpha,holes,ion,celec,dcelec,qdens=model_functions(NAcm,toxnm)
    M=N-1; Vgb=Vg-Vfb
    if guess is None: guess=1.0*np.exp(-x/6e-9)
    ph=guess[:M].copy()
    for _ in range(100):
        full=np.zeros(N); full[:M]=ph; nn=celec(full)[:M]; pp=holes(full)[:M]; naa=ion(full)[:M]
        dnn=dcelec(full)[:M]; dpp=-pp/VT; occ=np.clip(naa/NA,0,1); dna=NA*occ*(1-occ)/VT
        R=np.zeros(M); J=lil_matrix((M,M))
        R[0]=(-3*ph[0]+4*ph[1]-ph[2])/(2*dx)-alpha*(ph[0]-Vgb)
        J[0,0]=-3/(2*dx)-alpha; J[0,1]=4/(2*dx); J[0,2]=-1/(2*dx)
        for i in range(1,M):
            pip1=ph[i+1] if i+1<M else 0
            R[i]=(ph[i-1]-2*ph[i]+pip1)/dx**2-q/eps_si*(nn[i]+naa[i]-pp[i])
            J[i,i-1]=1/dx**2; J[i,i]=-2/dx**2-q/eps_si*(dnn[i]+dna[i]-dpp[i])
            if i+1<M: J[i,i+1]=1/dx**2
        delta=spsolve(J.tocsr(),-R); md=np.max(np.abs(delta))
        if md>0.1: delta*=0.1/md
        ph += 0.55*delta
        if np.max(np.abs(.55*delta))<2e-6: break
    out=np.zeros(N); out[:M]=ph
    return out,celec(out),(NA,EfEv,EcEf,p0,n0,alpha,holes,ion,celec,dcelec,qdens)

def solve_quantum_gate(NAcm,toxnm,Vg=Vg_map):
    pc,nc,mod=solve_classical_gate(NAcm,toxnm,Vg)
    NA,EfEv,EcEf,p0,n0,alpha,holes,ion,celec,dcelec,qdens=mod; Vgb=Vg-Vfb
    count=[0]
    def res(phi):
        count[0]+=1; nq=qdens(np.asarray(phi)); raw=solve_poisson_fixed_n(nq,np.asarray(phi),alpha,Vgb,holes,ion)
        return raw-np.asarray(phi)
    try:
        pq=anderson(res,pc,M=5,maxiter=45,f_tol=2e-4,w0=0.03,verbose=False)
    except Exception as e:
        # NoConvergence stores last iterate as first arg in SciPy
        try: pq=np.asarray(e.args[0],float)
        except Exception: pq=pc.copy()
    nq=qdens(pq)
    return pc,nc,pq,nq,mod,count[0]

def metrics_gate(NAcm,toxnm,Vg=Vg_map):
    pc,nc,pq,nq,mod,nev=solve_quantum_gate(NAcm,toxnm,Vg)
    n0=mod[4]
    exc_c=np.maximum(nc-n0,0); exc_q=np.maximum(nq-n0,0)
    Ns_c=np.trapezoid(exc_c,x); Ns_q=np.trapezoid(exc_q,x)
    xc=np.trapezoid(x*exc_c,x)/max(Ns_c,1e-300); xq=np.trapezoid(x*exc_q,x)/max(Ns_q,1e-300)
    dQ=100*abs(Ns_q-Ns_c)/max(Ns_c,1e-300)
    dXrel=100*abs(xq-xc)/max(xc,1e-12)
    dX=100*abs(xq-xc)/(toxnm*1e-9)  # centroid displacement normalized by physical oxide thickness
    return dQ,dX,dXrel,Ns_c/1e4,Ns_q/1e4,xc*1e9,xq*1e9,pq[0],pc[0],nev

# Threshold voltage via conventional surface-potential criterion phi_s=2 phi_F.
# At fixed phi_s the semiconductor solution depends on NA but not tox; tox only enters Vg=Vfb+phi_s-eps_si/Cox*dphi/dx|0.
def threshold_for_NA(NAcm):
    NA,EfEv,EcEf,p0,n0=setup_bulk(NAcm); phiF=VT*np.log(max(NAcm,1)/max(np.sqrt(Nc*Nv)*np.exp(-Eg/(2*VT))/1e6,1e-300))
    phis=2*phiF
    # model with arbitrary tox (not used in Dirichlet semiconductor solve)
    mod=model_functions(NAcm,3.0); NA,EfEv,EcEf,p0,n0,alpha,holes,ion,celec,dcelec,qdens=mod
    M=N-2
    def pois_dir(nfix,guess):
        ph=guess[1:-1].copy()
        for _ in range(80):
            full=np.empty(N); full[0]=phis; full[-1]=0; full[1:-1]=ph
            pp=holes(full)[1:-1]; naa=ion(full)[1:-1]; dpp=-pp/VT; occ=np.clip(naa/NA,0,1); dna=NA*occ*(1-occ)/VT
            R=np.zeros(M); J=lil_matrix((M,M))
            for kk,i in enumerate(range(1,N-1)):
                pim1=phis if i==1 else ph[kk-1]; pi=ph[kk]; pip1=0 if i==N-2 else ph[kk+1]
                R[kk]=(pim1-2*pi+pip1)/dx**2-q/eps_si*(nfix[i]+naa[kk]-pp[kk])
                if kk>0: J[kk,kk-1]=1/dx**2
                J[kk,kk]=-2/dx**2-q/eps_si*(dna[kk]-dpp[kk])
                if kk<M-1: J[kk,kk+1]=1/dx**2
            du=spsolve(J.tocsr(),-R); md=np.max(abs(du));
            if md>0.1: du*=0.1/md
            ph += .5*du
            if np.max(abs(.5*du))<2e-6: break
        out=np.empty(N); out[0]=phis; out[-1]=0; out[1:-1]=ph; return out
    # classical Dirichlet
    guess=phis*np.exp(-x/6e-9); ph=guess.copy()
    for _ in range(60):
        nn=celec(ph); new=pois_dir(nn,ph)
        if np.max(abs(new-ph))<2e-5: ph=new; break
        ph=.5*new+.5*ph
    pc=ph
    # quantum Anderson fixed point
    def r(phi): return pois_dir(qdens(np.asarray(phi)),np.asarray(phi))-np.asarray(phi)
    try: pq=anderson(r,pc,M=5,maxiter=40,f_tol=2e-4,w0=.03)
    except Exception as e:
        try: pq=np.asarray(e.args[0],float)
        except: pq=pc.copy()
    # surface derivative (positive x into Si)
    dc=(-3*pc[0]+4*pc[1]-pc[2])/(2*dx); dq=(-3*pq[0]+4*pq[1]-pq[2])/(2*dx)
    # Vg = Vfb + phis - (eps_si/Cox)*phi'(0)
    return phis,dc,dq

def worker(args):
    na,tox,Vg=args
    try:
        vals=metrics_gate(na,tox,Vg); return (na,tox,Vg,*vals,'ok')
    except Exception as e:
        return (na,tox,Vg,*([np.nan]*10),str(e))

if __name__=='__main__':
    NAvals=np.array([1e16,3e16,1e17,3e17,1e18,3e18],float)
    toxvals=np.array([1.5,2.0,3.0,5.0,8.0,12.0],float)

    # First solve the surface-potential-defined threshold state for each doping.
    # The threshold criterion used for the map is phi_s = 2 phi_F.  The resulting
    # semiconductor fields are then translated to V_T for every oxide thickness.
    th={}
    with ProcessPoolExecutor(max_workers=4) as ex:
        fs={ex.submit(threshold_for_NA,na):na for na in NAvals}
        for f in as_completed(fs):
            th[fs[f]]=f.result(); print('threshold NA',fs[f],th[fs[f]],flush=True)

    Vtc_grid=np.zeros((len(NAvals),len(toxvals)))
    Vtq_grid=np.zeros_like(Vtc_grid); dVT=np.zeros_like(Vtc_grid)
    for i,na in enumerate(NAvals):
        phis,dc,dq=th[na]
        for j,toxnm in enumerate(toxvals):
            Cox=eps_ox/(toxnm*1e-9)
            Vtc=Vfb+phis-(eps_si/Cox)*dc
            Vtq=Vfb+phis-(eps_si/Cox)*dq
            Vtc_grid[i,j]=Vtc; Vtq_grid[i,j]=Vtq
            dVT[i,j]=100*abs(Vtq-Vtc)/max(abs(Vtc),0.05)

    # Compare Qinv and charge centroid at a common operating condition:
    # one volt above the *classical* threshold at each (NA,tox) point.
    tasks=[]
    for i,na in enumerate(NAvals):
        for j,tox in enumerate(toxvals):
            tasks.append((na,tox,Vtc_grid[i,j]+1.0))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs=[ex.submit(worker,t) for t in tasks]
        for f in as_completed(futs):
            r=f.result(); rows.append(r); print('done',r[0],r[1],'Vg',r[2],'dQ,dXtox=',r[3],r[4],r[-1],flush=True)

    dQ=np.full((len(NAvals),len(toxvals)),np.nan)
    dX=np.full_like(dQ,np.nan); dXrel=np.full_like(dQ,np.nan)
    extra={}
    for r in rows:
        na,tox=r[0],r[1]; i=np.where(NAvals==na)[0][0]; j=np.where(toxvals==tox)[0][0]
        dQ[i,j]=r[3]; dX[i,j]=r[4]; dXrel[i,j]=r[5]; extra[(na,tox)]=r

    csvpath='/mnt/data/qm_onset_map_data.csv'
    with open(csvpath,'w',newline='') as fh:
        w=csv.writer(fh)
        w.writerow(['NA_cm-3','tox_nm','Vg_eval_V','Vt_classical_V','Vt_quantum_V','dQinv_percent','dVt_percent','centroid_shift_over_tox_percent','centroid_relative_change_percent','Ns_classical_cm-2','Ns_quantum_cm-2','centroid_classical_nm','centroid_quantum_nm','phi_s_quantum_V','phi_s_classical_V'])
        for i,na in enumerate(NAvals):
            for j,tox in enumerate(toxvals):
                r=extra[(na,tox)]
                # r: na,tox,Vg,dQ,dX,dXrel,Nsc,Nsq,xc,xq,phisq,phisc,nev,status
                w.writerow([na,tox,r[2],Vtc_grid[i,j],Vtq_grid[i,j],dQ[i,j],dVT[i,j],dX[i,j],dXrel[i,j],r[6],r[7],r[8],r[9],r[10],r[11]])

    X,Y=np.meshgrid(toxvals,NAvals)
    def mapplot(Z,title,cbar,out):
        fig,ax=plt.subplots(figsize=(7.2,5.4))
        zlo=np.nanmin(Z); zhi=np.nanmax(Z)
        levels=np.linspace(zlo,zhi,20) if zhi>zlo else np.linspace(0,1,20)
        cf=ax.contourf(X,Y,Z,levels=levels,extend='both')
        requested=[1,5,10]
        available=[v for v in requested if zlo<=v<=zhi]
        if available:
            cs=ax.contour(X,Y,Z,levels=available,colors='k',linewidths=1.5)
            ax.clabel(cs,fmt=lambda v:f'{v:.0f}%',fontsize=9)
        missing=[v for v in requested if v not in available]
        if missing:
            if zlo>10:
                note='Entire computed domain >10%'
            elif zhi<1:
                note='Entire computed domain <1%'
            else:
                note='Requested levels outside range: '+', '.join(f'{v}%' for v in missing)
            ax.text(0.02,0.02,note,transform=ax.transAxes,fontsize=8.5,va='bottom',ha='left',bbox=dict(boxstyle='round,pad=0.25',facecolor='white',alpha=0.8,edgecolor='0.5'))
        ax.set_yscale('log'); ax.set_xlabel(r'Oxide thickness $t_{ox}$ (nm)'); ax.set_ylabel(r'Acceptor concentration $N_A$ (cm$^{-3}$)')
        ax.set_title(title); cb=fig.colorbar(cf,ax=ax); cb.set_label(cbar); fig.tight_layout(); fig.savefig(out,dpi=240); plt.close(fig)
    mapplot(dQ,r'QM onset: inversion-charge correction at $V_G=V_{T,cl}+1$ V',r'$|Q_{inv}^{QM}-Q_{inv}^{cl}|/|Q_{inv}^{cl}|$ (%)','/mnt/data/qm_onset_Qinv.png')
    mapplot(dVT,r'QM onset: threshold-voltage correction ($\phi_s=2\phi_F$ criterion)',r'$|V_{th}^{QM}-V_{th}^{cl}|/|V_{th}^{cl}|$ (%)','/mnt/data/qm_onset_Vt.png')
    mapplot(dX,r'QM onset: charge-centroid displacement at $V_G=V_{T,cl}+1$ V',r'$|\bar{x}_{QM}-\bar{x}_{cl}|/t_{ox}$ (%)','/mnt/data/qm_onset_centroid.png')
    print('ranges dQ',np.nanmin(dQ),np.nanmax(dQ),'dVT',np.nanmin(dVT),np.nanmax(dVT),'dX/tox',np.nanmin(dX),np.nanmax(dX))
