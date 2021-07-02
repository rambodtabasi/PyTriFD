#! /usr/bin/env python
# -*- coding: utf-8 -*-
from PyTriFD import FD
import matplotlib.pyplot as plt

class TwoDimStokes(FD):

    def parse_additional_inputs(self):

        self.nu = self.inputs['material parameters']['nu']
        self.E = self.inputs['material parameters']['E']

        E = self.E
        nu = self.nu

        self.lambda_plus_2G = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.lambda_ = nu / (1.0 - nu)

        return

    def residual_operator(self, my_field_overlap_sorted):

        ux = my_field_overlap_sorted[0]
        uy = my_field_overlap_sorted[1]

        dx = self.deltas[0]
        dy = self.deltas[1]

        strain_xx = (ux[1:, 1:] - ux[1:, :-1]) / dx
        strain_yy = (uy[1:, 1:] - uy[:-1, 1:]) / dy
        strain_xy = 0.5 * ((ux[1:, 1:] - ux[:-1, 1:]) / dy +
                           (uy[1:, 1:] - uy[1:, :-1]) / dx)

        stress_xx = self.lambda_plus_2G * strain_xx + self.lambda_ * strain_yy
        stress_yy = self.lambda_plus_2G * strain_yy + self.lambda_ * strain_xx
        stress_xy = self.E / (1 + self.nu) * strain_xy

        residual1 = ((stress_xx[:-1, :-1] - stress_xx[:-1, 1:]) / dx +
                     (stress_xy[:-1, :-1] - stress_xy[1:, :-1]) / dy)

        residual2 = ((stress_xy[:-1, :-1] - stress_xy[:-1, 1:]) / dx +
                     (stress_yy[:-1, :-1] - stress_yy[1:, :-1]) / dy)


        return residual1, residual2


    def plot_solution(self):

        nodes = self.get_nodes_on_rank0()
        u = self.get_solution_on_rank0()
        u = u[0::2]

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
            fig.colorbar(p)
            plt.show()


if __name__ == "__main__":

    problem = TwoDimStokes('inputs.yml')
    problem.solve()
    # problem.plot_solution()

