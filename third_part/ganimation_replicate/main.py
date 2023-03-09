"""
Created on Dec 13, 2018
@author: Yuedong Chen
"""

from options import Options
from solvers import create_solver




if __name__ == '__main__':
    opt = Options().parse()

    solver = create_solver(opt)
    solver.run_solver()

    print('[THE END]')