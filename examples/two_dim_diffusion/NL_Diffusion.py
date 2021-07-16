#! /usr/bin/env python
# -*- coding: utf-8 -*-
from PyTriFD import FD
import numpy.ma as ma
import numpy as np
import matplotlib.pyplot as plt
import time as ttt
import warnings

warnings.filterwarnings("ignore")

class TwoDimDiffusion(FD):

    def residual_operator(self, my_field_overlap):

        u = my_field_overlap[::self.nodal_dofs]
        v = my_field_overlap[1::self.nodal_dofs]
        #p = my_field_overlap[2::self.nodal_dofs]

        ### Current solutions from previsou step
        u_n = self.solution_n[::2]
        v_n = self.solution_n[1::2]



        residual = np.zeros_like(my_field_overlap[:self.num_owned])
        #print ("entered compute residual")

        residual_x = residual[::self.nodal_dofs]
        residual_y = residual[1::self.nodal_dofs]


        dx2 = self.deltas[0] ** 2
        dy2 = self.deltas[1] ** 2

        u_state = ma.masked_array(u[self.my_neighbors]
            -u[:self.num_owned_neighb,None], mask=self.my_neighbors.mask)
        laplacian = self.ref_mag_state_invert * u_state
        laplacian_integ = (laplacian * self.my_volumes[self.my_neighbors]).sum(axis=1)

        grad_u = self.my_ref_pos_state_x * self.ref_mag_state_invert * u_state
        grad_u_integ = (grad_u * self.my_volumes[self.my_neighbors]).sum(axis=1)

        #residual_x[:] = (u_n-u[:self.num_owned_neighb])/self.time_step - grad_u_integ -(1/20000)* laplacian_integ
        residual_x[:] = laplacian_integ

        v_state = ma.masked_array(v[self.my_neighbors]
            -v[:self.num_owned_neighb,None], mask=self.my_neighbors.mask)
        laplacian = self.ref_mag_state_invert * v_state
        laplacian_integ = (laplacian * self.my_volumes[self.my_neighbors]).sum(axis=1)

        residual_y[:] = laplacian_integ

        residual[::self.nodal_dofs] = residual_x[:]
        residual[1::self.nodal_dofs] = residual_y[:]
        #residual[2::self.nodal_dofs] = p[:self.num_owned_neighb]

        #print ("exiting compute residual")
        return residual

    def plot_solution(self):

        nodes = self.get_nodes_on_rank0()
        all_variables = self.get_solution_on_rank0()
        u = all_variables[::self.nodal_dofs]
        v = all_variables[1::self.nodal_dofs]
        #pressure = all_variables[2::self.nodal_dofs]

        bounds = \
            self.inputs['discretization']['tensor product grid']['bounds']
        delta = \
            self.inputs['discretization']['tensor product grid']['delta']

        xmin, xmax = bounds[0]
        xnodes = int((xmax - xmin) / delta[0]) + 1



        if self.rank == 0:
            fig, ax = plt.subplots()
            p = ax.contourf(nodes[0].reshape(-1,xnodes),
                            nodes[1].reshape(-1,xnodes),
                            u.reshape(-1,xnodes))
            ax.set_aspect('equal')
            fig.colorbar(p)
            plt.show()

        if self.rank == 0:
            fig, ax = plt.subplots()
            p = ax.contourf(nodes[0].reshape(-1,xnodes),
                            nodes[1].reshape(-1,xnodes),
                            v.reshape(-1,xnodes))
            ax.set_aspect('equal')
            fig.colorbar(p)
            plt.show()


if __name__ == "__main__":

    problem = TwoDimDiffusion('inputs.yml')
    problem.solve()
    problem.plot_solution()

