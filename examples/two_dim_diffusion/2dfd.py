#! /usr/bin/env python
# -*- coding: utf-8 -*-
from PyTriFD import FD
import matplotlib.pyplot as plt

class TwoDimDiffusion(FD):

    def residual_operator(self, my_field_overlap_sorted):

        u = my_field_overlap_sorted[0]

        dx2 = self.deltas[0] ** 2
        dy2 = self.deltas[1] ** 2

        residual = (u[:-2, 1:-1]  / dx2 + u[1:-1, :-2] / dy2
                    - 2 * u[1:-1, 1:-1] / dx2 - 2 * u[1:-1, 1:-1] / dy2
                    + u[1:-1, 2:] / dy2 + u[2:, 1:-1] / dx2)


        return residual

    def plot_solution(self):

        nodes = self.get_nodes_on_rank0()
        u = self.get_solution_on_rank0()

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

    problem = TwoDimDiffusion('inputs.yml')
    problem.solve()
    problem.plot_solution()

