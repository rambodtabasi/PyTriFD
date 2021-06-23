#! /usr/bin/env python
from PyTriFD import FD
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

    def plot_solution(self):

        nodes = self.get_nodes_on_rank0()
        u = self.get_solution_on_rank0()

        if self.rank == 0:
            fig, ax = plt.subplots()
            ax.plot(nodes[0], u)
            plt.show()


if __name__ == "__main__":

    problem = OneDimNonlinearDiffusion('inputs.yml')
    problem.solve()
    u = problem.plot_solution()
