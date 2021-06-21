#! /usr/bin/env python
# -*- coding: utf-8 -*-

from PyTriFD import FD
from PyTrilinos import Epetra
import matplotlib.pyplot as plt


class OneDimNonlinearDiffusion(FD):

    def parse_additional_inputs(self):

        self.k = self.inputs['material parameters']['k']

        return

    def residual_operator(self, my_field_overlap_sorted):

        u = my_field_overlap_sorted[0]

        residual = ((u[:-2] - 2*u[1:-1] + u[2:]) /
                    (self.deltas[0] ** 2.0) - self.k * u[1:-1] * u[1:-1])

        return residual

    def get_nodes(self):

        balanced_map = self.my_nodes.Map()

        if self.rank == 0:
            my_global_elements = balanced_map.NumGlobalElements()
        else:
            my_global_elements = 0

        temp_map = Epetra.Map(-1, my_global_elements, 0, self.comm)
        nodes = Epetra.Vector(temp_map)

        importer = Epetra.Import(temp_map, balanced_map)

        nodes.Import(self.my_nodes, importer, Epetra.Insert)

        return nodes.ExtractCopy()

    def plot_solution(self):

        nodes = self.get_nodes()
        u = self.get_final_solution()

        if self.rank == 0:
            fig, ax = plt.subplots()
            ax.plot(nodes, u)
            plt.show()


if __name__ == "__main__":

    problem = OneDimNonlinearDiffusion('inputs.yml')
    problem.solve_one_step()
    u = problem.plot_solution()
