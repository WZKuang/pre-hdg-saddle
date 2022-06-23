# pre-hdg-saddle
Robust block diagonal preconditioner for condensed Hdiv-HDG schemes for saddle point problems.

Paper DOI: 10.1093/imanum/drac021

Numerical experiment code for the generalized Stokes problems, steady/unsteady linear elasticity problems.

## Requirements:
+ Netgen/NGSolve, version: 6.2.2006-100-gdc536cb, website: ngsolve.org.
+ ngs-petsc, website:https://github.com/NGSolve/ngs-petsc.

## Files:
+ krylovspace.py: Replacing the file in the ngsolve installation directory .../lib/.../ngsolve/krylovspace.py. One hack in the MinRes solver to deal with the BC.
+ cavityHDG.py: Num experiments for the generealized Stokes equations in the lid-driven cavity.
+ backHDG.py: Num experiments for the generealized Stokes equations in the backward-facing flow.
+ cavityLSHDG.py: Num experiments for the steady linear elasticity equations.
+ cavityLSHDG_unsteady.py: Num experiments for the unsteady linear elasticity equations.

