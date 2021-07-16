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

        p = my_field_overlap[::self.nodal_dofs]
        s = my_field_overlap[1::self.nodal_dofs]
        #p = my_field_overlap[2::self.nodal_dofs]

        ### Current solutions from previsou step
        p_n = self.solution_n[::2]
        s_n = self.solution_n[1::2]

        residual = np.zeros_like(my_field_overlap[:self.num_owned])
        #print ("entered compute residual")

        residual_eq1 = residual[::self.nodal_dofs]
        residual_eq2 = residual[1::self.nodal_dofs]

        dx2 = self.deltas[0] ** 2
        dy2 = self.deltas[1] ** 2

        ## calculation of viscosity based on current concentration distribution
        ones = np.ones_like(s)
        invert_visc = (np.exp(3.0*(ones-s)))**-1


        p_state = ma.masked_array(p[self.my_neighbors]
            -p[:self.num_owned_neighb,None], mask=self.my_neighbors.mask)
        s_state = ma.masked_array(s[self.my_neighbors]
            -s[:self.num_owned_neighb,None], mask=self.my_neighbors.mask)

        # gradient term in x-direction
        x_grad_p = self.gamma_p * self.omega * self.my_ref_pos_state_x * self.ref_mag_state_invert * p_state
        x_grad_p_integ = (x_grad_p * self.my_volumes[self.my_neighbors]).sum(axis=1)
        x_grad_s = self.gamma_p * self.omega * self.up_wind_indicator * self.my_ref_pos_state_x * self.ref_mag_state_invert * s_state
        x_grad_s_integ = (x_grad_s * self.my_volumes[self.my_neighbors]).sum(axis=1)
        x_grad_p_grad_s = x_grad_p_integ * x_grad_s_integ

        ## gradient terms in y
        y_grad_p =  self.gamma_p * self.omega * self.my_ref_pos_state_y * self.ref_mag_state_invert * p_state
        y_grad_p_integ = (y_grad_p * self.my_volumes[self.my_neighbors]).sum(axis=1)
        y_grad_s =  self.gamma_p * self.omega * self.up_wind_indicator * self.my_ref_pos_state_y * self.ref_mag_state_invert * s_state
        y_grad_s_integ = (y_grad_s * self.my_volumes[self.my_neighbors]).sum(axis=1)
        y_grad_p_grad_s = y_grad_p_integ * y_grad_s_integ


        grad_terms = y_grad_p_grad_s + x_grad_p_grad_s

        laplacian_p = self.gamma_p * self.omega *  self.ref_mag_state_invert * p_state
        laplacian_p_integ = (laplacian_p * self.my_volumes[self.my_neighbors]).sum(axis=1)

        laplacian_s = self.gamma_p * self.omega *  self.ref_mag_state_invert * s_state
        laplacian_s_integ = (laplacian_s * self.my_volumes[self.my_neighbors]).sum(axis=1)

        term_2 = (invert_visc[:self.num_owned_neighb] * grad_terms) + (2.0/10000)* laplacian_s_integ

        residual_eq1[:] = (grad_terms *3.0 ) + (2.0 * laplacian_p_integ)
        residual_eq2[:] = (s[:self.num_owned_neighb]-s_n)/self.time_step-term_2



        residual[::self.nodal_dofs] = residual_eq1[:]
        residual[1::self.nodal_dofs] = residual_eq2[:]

        #print ("exiting compute residual")
        return residual

    def plot_solution(self):

        nodes = self.get_nodes_on_rank0()
        all_variables = self.get_solution_on_rank0()
        u = all_variables[::self.nodal_dofs]
        v = all_variables[1::self.nodal_dofs]
        v_on_proc = v[:self.num_owned]
        my_mid_nodes_min = np.where(self.my_nodes[1] > 0.49 )
        my_mid_nodes_max = np.where(self.my_nodes[1] < 0.51 )
        my_mid_nodes = np.intersect1d(my_mid_nodes_min,my_mid_nodes_max)
        p_mid = self.solution_n[::2][my_mid_nodes]
        s_mid = self.solution_n[1::2][my_mid_nodes]
        #s_min = v[my_mid_nodes]
        x_mid = self.my_nodes[0][my_mid_nodes]
        y_mid = self.my_nodes[1][my_mid_nodes]

        plt.plot(x_mid,p_mid)
        #plt.plot(x_mid,s_mid)
        plt.show()
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

