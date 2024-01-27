
Different from convex mpc solved by dense qp, the omega here means the anguler velocity in body frame. This leads to difference in the state matrix of dynamics.

We use single shooting and multiple shooting respectively to solve nonlinear mpc with CasADi. The solver is Ipopt, which uses iterior point methed(ipm) to solve nonlinear program directly. In contrast, acados uses the HPIPM, another qp solver using ipm, introduces sqp to convert complex problems to sub qps, is 10 time faster than CasADi in our scenarios.

In our example, the result is that multiple shooting is approximately 10 time faster than single shooting.
# nmpc_casadi
