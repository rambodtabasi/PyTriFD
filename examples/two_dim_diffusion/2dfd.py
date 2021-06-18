#! /usr/bin/env python
# -*- coding: utf-8 -*-

from PyTriFD import FD

class TwoDimDiffusion(FD):

    def residual_operator(self, my_field_overlap_sorted):

        u = my_field_overlap_sorted[0]

        residual = ((u[:-2, :] + u[:, :-2] - 4*u[1:-1, 1:-1] + u[:, 2:] + u[2:, :]) /
                    (self.deltas[0] ** 2.0))

        return residual.flatten()


if __name__ == "__main__":

    problem = OneDimNonlinearDiffusion('one_dim_nonlinear_diffusion_inputs.yml')
    problem.solve_one_step()
    u = problem.get_final_solution()

    if problem.comm.MyPID() == 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(u)
        plt.show()
