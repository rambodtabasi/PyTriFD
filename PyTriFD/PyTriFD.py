#! /usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import os

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

from ensight import Ensight

from PyTrilinos import Epetra
from PyTrilinos import Teuchos
from PyTrilinos import Isorropia
from PyTrilinos import NOX


class FD(NOX.Epetra.Interface.Required,
         NOX.Epetra.Interface.Jacobian):
    """
       Class that inherits from `NOX.Epetra.Interface.Required
       <http://trilinos.sandia.gov/packages/pytrilinos/development/NOX.html>`_
       to produce the problem interface to NOX for solving equations with a
       finite difference discretization
    """

    def __init__(self, inputs):
        """Instantiate the problem object"""
        NOX.Epetra.Interface.Required.__init__(self)
        NOX.Epetra.Interface.Jacobian.__init__(self)

        # stores input dictionary as class attribute, either read from a
        # yaml file or directly from a Python dictionary
        if isinstance(inputs, str):
            with open(inputs) as f:
                self.inputs = yaml.load(f, yaml.FullLoader)
        else:
            self.inputs = inputs


        #Epetra communicator attributes
        self.comm = Epetra.PyComm()
        self.rank = self.comm.MyPID()
        self.size = self.comm.NumProc()

        self.verbose = self.inputs.get('verbose', False)

        # Parse inputs
        self.parse_inputs()
        # Setup problem grid
        self.create_grid()
        # Find the global family array
        self.get_neighborhoods()
        # Initialize the neighborhood graph
        self.init_neighborhood_graph()
        # Load balance
        self.load_balance()
        # Initialize the output files
        self.init_output()
        # Initialize the field graph
        self.init_field_graph()
        # Initialize grid data structures
        self.init_grid_data()
        # Get boundary condition info
        self.parse_boundary_conditions()
        # Finalize outputs
        self.finalize_output()

        return

    def debug_print(self, print_str:str, *vargs):
        print(f'proc {self.rank}: {print_str}', *vargs)
        return

    def parse_inputs(self):
        """Parse input parameters"""

        discretization_parameters = self.inputs['discretization']
        self.horizon = discretization_parameters.get('horizon', 1.1)
        self.deltas = \
            np.array(discretization_parameters['tensor product grid']['delta'],
                     dtype=np.double)

        self.output_parameters = self.inputs['output']

        self.boundary_conditions = self.inputs.get('boundary conditions', None)

        dofs = self.inputs.get('degrees of freedom', ['u'])
        self.dof_map = dict(zip(dofs, range(len(dofs))))
        self.nodal_dofs = len(self.dof_map)

        self.parse_additional_inputs()
        return

    def parse_additional_inputs(self):
        pass

    def get_lids_from_box(self, dof, box):

        nodes = self.my_nodes[:]
        lids = np.arange(nodes.shape[1], dtype=np.int32)
        mask = np.ones_like(lids, dtype=np.bool)

        for (dim, (dim_min, dim_max)) in enumerate(box):
            mask *= (nodes[dim] > dim_min) * (nodes[dim] < dim_max)

        return self.nodal_dofs * lids[mask] + dof

    def get_lids_from_point_radius(self, dof, point, radius):

        nodes = self.my_nodes[:]
        lids = np.arange(nodes.shape[1], dtype=np.int32)

        #Get lids
        self.debug_print('point', point)
        lids = self.my_tree.query_ball_point(point, r=radius, eps=0.0, p=2)

        return self.nodal_dofs * np.array(lids, np.int32) + dof


    def parse_boundary_conditions(self):

        self.dirichlet_bc_lids = []
        self.dirichlet_bc_values = []
        for bc in self.boundary_conditions:
            dof = bc.get('degree of freedom', 0)
            if isinstance(dof, str):
                dof = self.dof_map[dof]
            if bc['type'] == 'dirichlet':
                if 'box' in bc['region']:
                    (self.dirichlet_bc_lids
                     .append(self.get_lids_from_box(dof, bc['region']['box'])))
                elif 'point-radius' in bc['region']:
                    (self.dirichlet_bc_lids
                     .append(self.get_lids_from_point_radius(dof,
                        *list(bc['region']['point-radius'].values()))))
                self.dirichlet_bc_values.append(bc['value'])

    def init_output(self):
        """Initial Ensight file for simulation results"""
        viz_path = self.output_parameters.get('path',
                                              os.environ.get('VIZ_PATH', None))
        scalar_variables = self.output_parameters.get('scalar', None)
        vector_variables = self.output_parameters.get('vector', None)
        self.outfile = Ensight('output', vector_variables, scalar_variables,
                               self.comm, viz_path=viz_path)
        return

    def finalize_output(self):

        self.outfile.write_case_file(self.comm)
        return

    def create_grid(self):
        """Creates initial rectangular grid"""
        if self.rank == 0:
            bounds = \
                self.inputs['discretization']['tensor product grid']['bounds']
            delta = \
                self.inputs['discretization']['tensor product grid']['delta']
            grid_linear_spaces = [np.arange(bounds[0], bounds[1] +
                                            delta, delta) for bounds, delta
                                  in zip(bounds, delta)]
            grid = np.meshgrid(*grid_linear_spaces)

            #create x, xy, xyz unraveled grid
            nodes = np.array([arr.flatten() for arr in grid],
                                  dtype=np.double).T

            my_num_nodes = nodes.shape[0]
            num_dims = nodes.shape[1]

        else:
            nodes = np.array([],dtype=np.double)
            my_num_nodes = nodes.shape[0]
            num_dims = 0

        self.global_number_of_nodes = self.comm.SumAll(my_num_nodes)
        self.problem_dimension = self.comm.SumAll(num_dims)

        # Create a temporary unbalanced map
        unbalanced_map = Epetra.Map(self.global_number_of_nodes,
                                    nodes.shape[0], 0,
                                    self.comm)

        # Create the unbalanced Epetra vector that will store the nodes
        self.my_nodes = Epetra.MultiVector(unbalanced_map,
                                           self.problem_dimension)
        self.my_nodes[:] = nodes.T

        return

    def get_neighborhoods(self):
        """cKDTree implemented for neighbor search """

        if self.rank == 0:
            #Create a kdtree to do nearest neighbor search
            tree = scipy.spatial.cKDTree(self.my_nodes[:].T)

            #Get all neighborhoods
            self.neighborhoods = tree.query_ball_point(self.my_nodes[:].T,
                                                       r=self.horizon,
                                                       eps=0.0, p=2)
        else:
            #Setup empty data on other ranks
            self.neighborhoods = []

        return

    def init_neighborhood_graph(self):
        """
           Creates the neighborhood ``connectivity'' graph.  This is used to
           load balanced the problem and initialize Jacobian data.
        """

        #Create the standard unbalanced map to instantiate the Epetra.CrsGraph
        #This map has all nodes on the 0 rank processor.
        unbalanced_map = self.my_nodes.Map()

        #Compute a list of the lengths of each neighborhood list
        num_indices_per_row = np.array([ len(item)
            for item in self.neighborhoods ], dtype=np.int32)

        #Instantiate the graph
        self.neighborhood_graph = Epetra.CrsGraph(Epetra.Copy, unbalanced_map,
                                                  num_indices_per_row, True)

        #Fill the graph
        for rid, row in enumerate(self.neighborhoods):
            self.neighborhood_graph.InsertGlobalIndices(rid,row)
        #Complete fill of graph
        self.neighborhood_graph.FillComplete()

        return

    def load_balance(self):
        """Load balancing function."""
        # Load balance
        if self.rank == 0:
            print("Load balancing...\n")
        # Create Teuchos parameter list to pass parameters to ZOLTAN for load
        # balancing
        parameter_list = Teuchos.ParameterList()
        parameter_list.set("Partitioning Method", "RCB")
        parameter_sublist = parameter_list.sublist("ZOLTAN")
        parameter_sublist.set("RCB_RECTILINEAR_BLOCKS", "1")
        if not self.verbose:
            parameter_sublist.set("DEBUG_LEVEL", "0")
        # Create a partitioner to load balance the graph
        partitioner = Isorropia.Epetra.Partitioner(self.my_nodes,
                                                   parameter_list)
        # And a redistributer
        redistributer = Isorropia.Epetra.Redistributor(partitioner)

        # Redistribute vector and store the map
        self.neighborhood_graph = \
            redistributer.redistribute(self.neighborhood_graph)

        self.balanced_map = self.neighborhood_graph.Map()

        return

    def init_field_graph(self):

        nodal_dofs = self.nodal_dofs

        my_global_indices = self.balanced_map.MyGlobalElements()

        field_global_indices = np.empty(nodal_dofs *
                                        my_global_indices.shape[0],
                                        dtype=np.int32)

        for i in range(nodal_dofs):
            field_global_indices[i::nodal_dofs] = (nodal_dofs *
                                                   my_global_indices) + i

        self.number_of_field_variables =  (nodal_dofs *
                                           self.global_number_of_nodes)

        # create Epetra Map based on node degrees of Freedom
        self.balanced_field_map = Epetra.Map(self.number_of_field_variables,
                                             field_global_indices.tolist(),
                                             0, self.comm)

        # Instantiate the corresponding graph
        self.balanced_field_graph = Epetra.CrsGraph(Epetra.Copy,
                                                    self.balanced_field_map,
                                                    True)
        # fill the field graph
        for i in my_global_indices:
            # array of global indices in neighborhood of each node
            global_index_array = (self.neighborhood_graph
                                      .ExtractGlobalRowCopy(i))
            # convert global node indices to appropriate field indices
            field_index_array = []
            for j in range(nodal_dofs):
                field_index_array.append(nodal_dofs * global_index_array + j)

            field_index_array = np.sort(np.array(field_index_array).flatten())

            # insert rows into balanced graph per appropriate rows
            for j in range(nodal_dofs):
                self.balanced_field_graph.InsertGlobalIndices(
                        nodal_dofs * i + j, field_index_array)

        # complete fill of balanced graph
        self.balanced_field_graph.FillComplete()
        self.num_owned = self.balanced_field_graph.NumMyRows()

        return


    def init_grid_data(self):
        """Create data structures needed for doing computations"""

        # Create some local (to function) convenience variables
        unbalanced_map = self.my_nodes.Map()
        balanced_map = self.balanced_map
        field_balanced_map = self.balanced_field_map

        overlap_map = self.neighborhood_graph.ColMap()
        field_overlap_map = self.balanced_field_graph.ColMap()


        # Create the balanced vectors
        my_nodes = Epetra.MultiVector(balanced_map,
                                      self.problem_dimension)
        self.my_field = Epetra.Vector(field_balanced_map)

        # Create the overlap vectors
        self.my_nodes_overlap = Epetra.MultiVector(overlap_map,
                                                   self.problem_dimension)
        self.my_field_overlap = Epetra.Vector(field_overlap_map)


        # Create/get importers
        grid_importer = Epetra.Import(balanced_map, unbalanced_map)
        grid_overlap_importer = Epetra.Import(overlap_map, balanced_map)

        self.field_overlap_importer = Epetra.Import(field_overlap_map,
                                                    field_balanced_map)

        # Import the unbalanced nodal data to balanced and overlap data
        my_nodes.Import(self.my_nodes, grid_importer, Epetra.Insert)
        self.my_nodes = my_nodes

        #Create a kdtree to do nearest neighbor searches
        self.my_tree = scipy.spatial.cKDTree(self.my_nodes[:].T)

        # Write the now balanced nodes to output file
        self.outfile.write_geometry_file_time_step(*self.my_nodes[:])
        np.save(f'proc_number_{self.comm.MyPID()}.npy', self.my_nodes[:])

        # Write the processor number if requested
        if 'processor_number' in self.output_parameters:
            proc_list = np.ones(self.my_nodes[:].T.shape[0],
                                dtype=np.int32) * self.comm.MyPID()
            self.outfile.write_scalar_variable_time_step('processor_number',
                                                         proc_list, 0.0)


        # Import the balanced nodal data to overlap data
        self.my_nodes_overlap.Import(my_nodes, grid_overlap_importer,
                                     Epetra.Insert)
        np.save(f'proc_number_overlap_{self.comm.MyPID()}.npy', self.my_nodes_overlap[:])

        maxes = np.max(self.my_nodes_overlap[:], axis=1)
        mins = np.min(self.my_nodes_overlap[:], axis=1)
        self.my_strides = \
            np.around((maxes - mins) / self.deltas + 1).astype(np.int)[::-1]

        overlap_indices = overlap_map.MyGlobalElements()
        field_overlap_indices = field_overlap_map.MyGlobalElements()

        self.my_overlap_indices_sorted = np.argsort(overlap_indices)
        self.my_field_overlap_indices_sorted = np.argsort(field_overlap_indices)
        self.my_field_overlap_indices_unsorted = \
            np.argsort(self.my_field_overlap_indices_sorted)

        self.F_fill = np.zeros_like(self.my_field_overlap[:])

        return

    def computeF(self, x, F, flag):
        """Implements the residual calculation as required by NOX.
        """
        try:
            num_owned = self.num_owned
            field_overlap_importer = self.field_overlap_importer

            # Ensure residual placeholder is 0.0 everywhere
            self.F_fill[:] = 0.0

            # Import off-processor data
            self.my_field_overlap.Import(x, field_overlap_importer,
                                         Epetra.Insert)

            # Sort data for FD calcs
            my_field_overlap_sorted = \
                (self.my_field_overlap[self.my_field_overlap_indices_sorted]
                 .reshape(self.nodal_dofs, *self.my_strides))

            # return the (sorted) residual
            self.F_fill[1:-1] = self.residual_operator(my_field_overlap_sorted)

            # Unsort and fill the actual residual
            F[:] = self.F_fill[self.my_field_overlap_indices_unsorted][:num_owned]

            for bc_lids, bc_values in zip(self.dirichlet_bc_lids,
                                          self.dirichlet_bc_values):
                F[bc_lids] = x[bc_lids] - bc_values

            self.my_field[:] = x

            return True

        except Exception as e:
            print ("Exception in PD.computeF method")
            print (e)

            return False

    def residual_operator(self, my_field_overlap_sorted):
        raise NotImplementedError()

    def solve_one_step(self):

        #Suppress 'Aztec status AZ_loss: loss of precision' messages
        self.comm.SetTracebackMode(0)

        #Get the initial solution vector
        initial_guess    = self.my_field
        nox_initial_guess = NOX.Epetra.Vector(initial_guess,
                                              NOX.Epetra.Vector.CreateView)

        # Define the ParameterLists
        nonlinear_parameters = \
            NOX.Epetra.defaultNonlinearParameters(self.comm, 2)
        print_parameters = nonlinear_parameters["Printing"]
        linear_solver_parameters = nonlinear_parameters["Linear Solver"]

        # Define the Jacobian interface/operator
        matrix_free_operator = NOX.Epetra.MatrixFree(print_parameters, self,
                                                     nox_initial_guess)

        # Define the Preconditioner interface/operator
        preconditioner = \
            NOX.Epetra.FiniteDifferenceColoring(print_parameters, self,
                                                initial_guess,
                                                self.balanced_field_graph,
                                                True)

        #Create and execute the solver
        solver = NOX.Epetra.defaultSolver(initial_guess, self,
                                          matrix_free_operator,
                                          matrix_free_operator,
                                          preconditioner, preconditioner,
                                          nonlinear_parameters)

        solve_status = solver.solve()

        if solve_status != NOX.StatusTest.Converged:
            if self.rank == 0:
                print("Nonlinear solver failed to converge")


    def get_final_solution(self):

        balanced_map = self.my_field.Map()

        if self.rank == 0:
            my_global_elements = balanced_map.NumGlobalElements()
        else:
            my_global_elements = 0

        temp_map = Epetra.Map(-1, my_global_elements, 0, self.comm)
        solution = Epetra.Vector(temp_map)

        importer = Epetra.Import(temp_map, balanced_map)

        solution.Import(self.my_field, importer, Epetra.Insert)

        return solution.ExtractCopy()
