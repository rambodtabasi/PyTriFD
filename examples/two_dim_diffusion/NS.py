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

        #print ("entered compute residual")
        u = my_field_overlap[::2]
        v = my_field_overlap[1::2]
        p = my_field_overlap[2::2]

        ### Solutions from the previsou step
        u_n = self.solution_n[:self.dofs_size]
        v_n = self.solution_n[self.dofs_size:2*self.dofs_size]
        #p_n = self.solution_n[2::self.nodal_dofs]

        residual = np.zeros_like(my_field_overlap[:self.num_owned])

        residual_eq1 = residual[:self.dofs_size]
        residual_eq2 = residual[self.dofs_size:2*self.dofs_size]
        residual_eq3 = residual[2*self.dofs_size:]
        print ("setup is ready")

        ## Material Properties
        ## at 15C
        rho = 1.225
        viscosity = 1.8e-5
        visc_sum = 2*(viscosity/rho)

        u_state = ma.masked_array(u[self.my_neighbors]
            -u[:self.num_owned_neighb,None], mask=self.my_neighbors.mask)
        v_state = ma.masked_array(v[self.my_neighbors]
            -v[:self.num_owned_neighb,None], mask=self.my_neighbors.mask)
        p_state = ma.masked_array(p[self.my_neighbors]
            -p[:self.num_owned_neighb,None], mask=self.my_neighbors.mask)
        print ("1")

        upwind_indicator = np.sign(self.my_ref_pos_state_x * u[:self.num_owned_neighb,np.newaxis] +self.my_ref_pos_state_y *\
                         v[:self.num_owned_neighb,np.newaxis])
        print ("2")

        ### equation 1, x-direction
        grad_u_x = self.gamma * self.omega * u_state * (self.my_ref_pos_state_x) * self.ref_mag_state_invert
        grad_u_x = np.where(upwind_indicator > 0, 0.0, grad_u_x)
        integ_grad_u_x = (grad_u_x * self.my_volumes[self.my_neighbors]).sum(axis=1)
        grad_u_y = self.gamma * self.omega * u_state * (self.my_ref_pos_state_y) * self.ref_mag_state_invert
        grad_u_y = np.where(upwind_indicator > 0, 0.0, grad_u_y)
        integ_grad_u_y = (grad_u_y * self.my_volumes[self.my_neighbors]).sum(axis=1)
        term_1_x = u[:self.num_owned_neighb] * integ_grad_u_x + v[:self.num_owned_neighb] * integ_grad_u_y

        laplace_u_x = self.beta * self.omega * visc_sum * (u_state)*(self.my_ref_pos_state_x **2)*(self.ref_mag_state_invert)
        term_2_x = (laplace_u_x * self.my_volumes[self.my_neighbors]).sum(axis=1)

        grad_p = self.gamma * self.omega * p_state * \
            (self.my_ref_pos_state_x) * self.ref_mag_state_invert
        term_3_x = (1/rho) * (grad_p * self.my_volumes[self.my_neighbors]).sum(axis=1)
        residual_eq1 = ((u[:self.num_owned_neighb] - u_n) / self.time_step) + term_3_x + term_1_x - term_2_x
        print ("residual 1 calculated")

        ### equation 2, y-direction
        grad_v_x = self.gamma * self.omega * v_state * (self.my_ref_pos_state_x) * self.ref_mag_state_invert
        grad_v_x = np.where(upwind_indicator > 0, 0.0, grad_v_x)
        integ_grad_v_x = (grad_v_x * self.my_volumes[self.my_neighbors]).sum(axis=1)
        grad_v_y = self.gamma * self.omega * v_state * (self.my_ref_pos_state_y) * self.ref_mag_state_invert
        grad_v_y = np.where(upwind_indicator > 0, 0.0, grad_v_y)
        integ_grad_v_y = (grad_v_y * self.my_volumes[self.my_neighbors]).sum(axis=1)
        term_1_y = u[:self.num_owned_neighb] * integ_grad_v_x + v[:self.num_owned_neighb] * integ_grad_v_y
        laplace_v_x = self.beta * self.omega * visc_sum * (v_state)*(self.my_ref_pos_state_y **2)*(self.ref_mag_state_invert)
        term_2_y = (laplace_v_x * self.my_volumes[self.my_neighbors]).sum(axis=1)
        grad_p = self.gamma * self.omega * p_state * \
            (self.my_ref_pos_state_x) * self.ref_mag_state_invert
        term_3_y = (1/rho) * (grad_p * self.my_volumes[self.my_neighbors]).sum(axis=1)
        residual_eq2 = ((v[:self.num_owned_neighb] - v_n) / self.time_step) + term_3_y + term_1_y - term_2_y
        print ("residual 2 calculated")



        ### equation 3, incompressibility
        grad_u_x = self.gamma * self.omega * u_state * (self.my_ref_pos_state_x) * self.ref_mag_state_invert
        integ_grad_u_x = (grad_u_x * self.my_volumes[self.my_neighbors]).sum(axis=1)
        grad_v_y = self.gamma * self.omega * v_state * (self.my_ref_pos_state_y) * self.ref_mag_state_invert
        integ_grad_v_y = (grad_v_y * self.my_volumes[self.my_neighbors]).sum(axis=1)
        residual_eq3 = integ_grad_u_x + integ_grad_v_y
        print ("residual 3 calculated")
        #print ("Calculation is complete")



        #print ("exiting compute residual")
        residual[:self.dofs_size] = residual_eq1[:]
        residual[self.dofs_size:2*self.dofs_size] = residual_eq2[:]
        residual[2*self.dofs_size:] = residual_eq3[:]
        return residual

    def plot_solution(self):

        nodes = self.get_nodes_on_rank0()
        all_variables = self.get_solution_on_rank0()
        nodes = self.get_nodes_on_rank0()
        u = all_variables[::self.nodal_dofs]
        v = all_variables[1::self.nodal_dofs]
        solution = self.get_solution()
        u_sol = solution[::2]
        v_sol=solution[1::2]
        my_x = nodes[::2][:self.num_owned_neighb]
        my_y = nodes[1::2][:self.num_owned_neighb]
        if self.rank == 0:
            plt.scatter(my_x,my_y,c=u)
            plt.show()
            plt.scatter(my_x,my_y,c=v)
            plt.show()

        """
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
        """

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
            plt.savefig("graphs/"+ "x_" + str(self.current_i)+".png")


        if self.rank == 0:
            fig, ax = plt.subplots()
            p = ax.contourf(nodes[0].reshape(-1,xnodes),
                            nodes[1].reshape(-1,xnodes),
                            v.reshape(-1,xnodes))
            ax.set_aspect('equal')
            fig.colorbar(p)
            plt.savefig("graphs/"+ "Y_" + str(self.current_i)+".png")


if __name__ == "__main__":

    problem = TwoDimDiffusion('inputs.yml')
    problem.solve()
    problem.plot_solution()

