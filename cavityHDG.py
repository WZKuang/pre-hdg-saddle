# Solves - Lap u + tau u - grad p = 1 on [0,1]^d
#                           div u = 0
# ngsolve-imports
from ngsolve.krylovspace import MinRes
from ngsolve.meshes import *
from ngsolve import *
from numpy import pi
# petsc solver
import ngs_petsc 
import time as timeit

ngsglobals.msg_level=0

def SolveProblem(order=1, refines=3, dim=3, tau=1, hd=True, red=True, 
        pr=False, draw=False):
    if draw:
        import netgen.gui
    for i in range(refines):
        t0 = timeit.time()
        if dim==2:
            nx = 8*2**i
            mesh = MakeStructured2DMesh(quads=False, nx=nx, ny = nx)
            utop = CoefficientFunction((4*x*(1-x),0))
        else:
            nx = 4*(i+1)
            mesh = MakeStructured3DMesh(hexes=False, nx=nx, ny = nx, nz = nx)
            utop = CoefficientFunction((16*x*(1-x)*y*(1-y),0,0))
        t1 = timeit.time()
        print("\nOrder: ", order, "LVL: ", i, " DIM: ", dim, " TAU: ", tau)
        print("projected_jumps: ", hd, "  ho_div_free: ", red)
        print("\nElasped:%.2e MESHING "%(t1-t0))
        
        # Div-HDG spaces
        V = HDiv(mesh, order=order, hodivfree=red, dirichlet=".*")
        VF = TangentialFacetFESpace(mesh, order=order, 
                highest_order_dc=hd, dirichlet=".*")
        if red: # P0 for pres
            W = L2(mesh, order=0, lowest_order_wb=True)
        else:
            W = L2(mesh, order=order-1, lowest_order_wb=True)

        fes = FESpace([V, VF, W])
        #fes.FreeDofs().Clear(V.ndof+VF.ndof)
        # aux H1 space
        V0 = VectorH1(mesh, order=1, dirichlet=".*")
        
        gfu = GridFunction (fes)
        uh, uhath, ph = gfu.components
        (u,uhat, p),  (v,vhat,q) = fes.TnT()
        u0, v0 = V0.TnT()

        # gradient by row
        #gradv, gradu = Grad(v), Grad(u)
        gradv, gradu = Sym(Grad(v)), Sym(Grad(u))
        
        # RHS (constant) 
        f = LinearForm (fes)

        # normal direction and mesh size
        n = specialcf.normal(mesh.dim)
        h = specialcf.mesh_size
        # stability parameter
        if dim ==2:
            alpha = 4*order**2/h
        else:
            alpha = 8*order**2/h

        # tangential component
        def tang(v):
            return v-(v*n)*n

        ########### HDG operator ah
        a = BilinearForm (fes, symmetric=True, condense=True)
        # volume term
        a += (2*InnerProduct(gradu,gradv)+tau*u*v
                -div(u)*q-div(v)*p)*dx
        # bdry terms
        a += 2*(-gradu*n*tang(v-vhat)-gradv*n*tang(u-uhat)
                +alpha*tang(u-uhat)*tang(v-vhat))*dx(element_boundary=True)

        
        ######## facet patch (edge path) for block smoother
        udofs = BitArray(fes.ndof)
        udofs[:] = 0
        udofs[:V.ndof] = V.FreeDofs(True)
        udofs[V.ndof:V.ndof+VF.ndof] = VF.FreeDofs(True)
        
        def edgePatchBlocks(mesh, fes):
            blocks = []
            freedofs = udofs
            for e in mesh.edges:
                edofs = set()
                # get ALL dofs connected to the edge
                for el in mesh[e].elements:
                    edofs |= set(d for d in fes.GetDofNrs(el)
                                 if freedofs[d])
                blocks.append(edofs)
            return blocks
        
        # face blocks for projector mass inversion
        def faceBlocks(mesh, fes):
            blocks = []
            freedofs = udofs
            for e in mesh.faces:
                # get ALL dofs connected to the edge
                fdofs = set(d for d in fes.GetDofNrs(e)
                             if freedofs[d])
                blocks.append(fdofs)
            return blocks
        
        eBlocks = edgePatchBlocks(mesh, fes)
        t0 = timeit.time()
        # number of DOFS
        ntotal = fes.ndof 
        nglobal = sum(fes.FreeDofs(True))
        print("Elasped:%.2e BLOCKING  Total DOFs: %.2e Global DOFs: %.2e "%(
            t0-t1, ntotal, nglobal))
        
        class SymmetricGS(BaseMatrix):
              def __init__ (self, smoother):
                  super(SymmetricGS, self).__init__()
                  self.smoother = smoother
              def Mult (self, x, y):
                  y[:] = 0.0
                  self.smoother.Smooth(y, x)
                  self.smoother.SmoothBack(y,x)
              def Height (self):
                  return self.smoother.height
              def Width (self):
                  return self.smoother.height
        
        
        ########### ASP operator ah0
        a0 = BilinearForm(V0)
        a0 += (InnerProduct(Grad(u0), Grad(v0))+tau*u0*v0)*dx
        
        # Projection operator V0--> fes
        # We set up a mixed mass matrix for V0 -> fes 
        # and then solving with the mass matrix in M
        mixmass = BilinearForm(trialspace=V0, testspace=fes)
        # tangential part
        mixmass += tang(u0) * tang(vhat) * dx(element_boundary=True)
        # normal part
        mixmass += (u0*n) * (v*n) * dx(element_boundary=True)
        
        massf = BilinearForm(fes)
        massf += tang(uhat) * tang(vhat) * dx(element_boundary=True)
        massf += (u*n) * (v*n) * dx(element_boundary=True)
        
        ####### pressure Schur complement part
        #### Prepare from W to piece-wise constant space W0
        pdofs = BitArray(fes.ndof)
        pdofs[:] = 0
        pdofs[V.ndof+VF.ndof:] = W.FreeDofs(True)

        W0 = L2(mesh, order=0, dgjumps=True)
        #W0.FreeDofs().Clear(0)
        p0,q0 = W0.TnT()
        
        p_mixmass = BilinearForm(trialspace=W0, testspace=fes)
        p_mixmass += p0*q*dx

        p_mass = BilinearForm(fes)
        p_mass += p*q*dx

        # M - mass matrix on W0(p0)
        M = BilinearForm(W0)
        M += p0*q0*dx
        
        N = BilinearForm(W0)
        N += 1/h*(p0-p0.Other())*(q0-q0.Other())*dx(skeleton=True)

        with TaskManager():
            f.Assemble()
            a.Assemble()
            t1 = timeit.time()
            print("Elasped:%.2e ASSEMBLE "%(t1-t0))
            ######### smoother (use edge blocks)
            bjac = a.mat.CreateBlockSmoother(eBlocks)

            ######## ASP
            a0.Assemble()
            pm = ngs_petsc.PETScMatrix(a0.mat, V0.FreeDofs())
            inva1 = ngs_petsc.PETSc2NGsPrecond(pm, 
                    petsc_options = {"pc_type": "hypre"})
            
            massf.Assemble()
            mixmass.Assemble()
            # massf is diagonal (only in 2D)!!!!!!!!!
            if dim==2:
                m_inv = massf.mat.CreateSmoother(udofs) 
            else: # TODO : block smoother
                fBlocks = faceBlocks(mesh, fes)
                m_inv = massf.mat.CreateBlockSmoother(fBlocks) 

            E = m_inv @ mixmass.mat
            ET = mixmass.mat.T @ m_inv
            pre_twogrid = E @ inva1 @ ET

            # block gs smoother
            preU = SymmetricGS(bjac) + pre_twogrid

            ### Schur Complement : Assemble M-1,N-1
            if red: # W0==W ==> only need embedding
                Ep = Embedding(fes.ndof, fes.Range(2))
                EpT = Ep.T
            else:
                p_mass.Assemble()
                p_mixmass.Assemble()
                p_mass_inv = p_mass.mat.CreateSmoother(pdofs) # orthogonal basis
                Ep = p_mass_inv @ p_mixmass.mat
                EpT = p_mixmass.mat.T @ p_mass_inv

            M.Assemble()
            N.Assemble()
            M_inv = M.mat.CreateSmoother(W0.FreeDofs())
            # no use in steady Stokes
            N_petscM = ngs_petsc.PETScMatrix(N.mat, W0.FreeDofs())
            N_inv = ngs_petsc.PETSc2NGsPrecond(N_petscM, 
                    petsc_options={"pc_type":"hypre"})
            if tau ==0: 
                preP = Ep @ M_inv @ EpT
            else:
                preP = Ep @ (2*M_inv+tau*N_inv) @ EpT

            PREC = preU + preP
            t2 = timeit.time()
            print("Elasped:%.2e PREC "%(t2-t1))
            
            ### Boundary condition data
            uhath.Set(utop, definedon=mesh.Boundaries("top"))
            # random initial value
            from scipy import random
            tmp = gfu.vec.CreateVector()
            tmp.FV().NumPy()[:] = random.rand(fes.ndof)
            gfu.vec.data += Projector(fes.FreeDofs(True), True) * tmp
            if draw:
                Draw(Norm(uh), mesh, "ini")
                input("ini")

            f.vec.data = -a.mat * gfu.vec

            f.vec.data += a.harmonic_extension_trans * f.vec
            # solver
            # Relative tol = 1e-8 or absolute tol = 1e-10
            error0 = sqrt(InnerProduct(PREC*f.vec,f.vec))
            tol = max(1e-10,1e-8*error0)
            _, it = MinRes(mat=a.mat, pre=PREC, rhs=f.vec,sol=gfu.vec,
                      initialize=False, maxsteps=500,printrates=pr, tol=tol)
            #gfu.vec.data += r
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * f.vec
            t3 = timeit.time()
            print("Elasped:%.2e SOLVE   BGS: %i "%(t3-t2, it)) 
            
            print("*****************************************************")
            print("*****************************************************")
            print("*****************************************************")
            if draw:
                Draw(Norm(uh), mesh, "vel")
                input("view")


############### parameters
dimList = [3]
orderList = [2,3,4]
tauList = [0, 1, 100]
refines = 4

for dim in dimList:
    for order in orderList:
        for tau in tauList:
            SolveProblem(order, refines, dim=dim, tau=tau, hd=True, red=True,
                    pr=False,draw=False)
