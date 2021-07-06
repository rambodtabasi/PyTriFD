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

        ux_xx = (ux[1:-1, :-2] - 2 * ux[1:-1, 1:-1] + ux[1:-1, 2:]) / dx / dx
        uy_xx = (uy[1:-1, :-2] - 2 * uy[1:-1, 1:-1] + uy[1:-1, 2:]) / dx / dx
        uy_yy = (uy[:-2, 1:-1] - 2 * uy[1:-1, 1:-1] + uy[2:, 1:-1]) / dy / dy
        ux_yy = (ux[:-2, 1:-1] - 2 * ux[1:-1, 1:-1] + ux[2:, 1:-1]) / dy / dy

        ux_xy = ((ux[2:, 2:] - ux[:-2, 2:] - ux[2:, :-2] + ux[:-2, :-2]) /
                 dx / dy / 4.0)
        uy_xy = ((uy[2:, 2:] - uy[:-2, 2:] - uy[2:, :-2] + uy[:-2, :-2]) /
                 dx / dy / 4.0)

        sxx_x = self.lambda_plus_2G * ux_xx + self.lambda_ * uy_xy
        syy_y = self.lambda_plus_2G * uy_yy + self.lambda_ * ux_xy

        sxy_y = self.E / (1.0 + self.nu) / 2.0 * (ux_yy + uy_xy)
        sxy_x = self.E / (1.0 + self.nu) / 2.0 * (ux_xy + uy_yy)

        residual1 = sxx_x + sxy_y
        residual2 = sxy_x + syy_y

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

