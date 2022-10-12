from ortools.algorithms import pywrapknapsack_solver
import time
import csv
import os
import random


def solve(values, weights, capacities):
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

    solver.Init(values, weights, capacities)
    solver.set_time_limit(180)
    computed_value = solver.Solve()

    packed_items = []
    packed_weights = []
    total_weight = 0
    # print('Total value = ', computed_value)
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
    # print('Total weight = ', total_weight)
    # print('Packed items:', packed_items)
    # print('Packed_weights:', packed_weights)
    return computed_value, total_weight


if __name__ == '__main__':
    
    g = open('ForExampleAnswer.csv', 'w')
    list_file = ['kplib/00Uncorrelated/n00050/R01000/s000.kp']

    # g = open('AnswerStatical.csv', 'w')
    # list_file = []

    # for test_type in os.listdir('kplib/'):
    #     if test_type[:2].isnumeric() == False:
    #         continue
        
    #     for test_num in os.listdir('kplib/' + test_type):
    #         list_test_r = ['R01000', 'R10000']
    #         test_r = list_test_r[random.randrange(2)]
    #         test_s = 's' + '{:03d}'.format(random.randrange(100)) + '.kp'
    #         list_file.append('kplib/' + test_type + '/' + test_num + '/' + test_r + '/' + test_s)

    writer = csv.writer(g)
    writer.writerow(['Type test', 'Num test item', 'Limit', 'File name', 'Total Value', 'Total Weight', 'Runtime', 'Is Optimal?'])        

    for file in list_file:

        f = open(file, 'r')

        f.readline()
        n = int(f.readline())
        capacity = int(f.readline())
        f.readline()

        values = []
        weights = []

        for line in f.readlines():
            values.append(int(line.split(' ')[0]))
            weights.append(int(line.split(' ')[1]))
    
        time_start = time.time()
        total_value, total_weight = solve(values, [weights] , [capacity])
        time_end = time.time()
        run_time = float('{:.5f}'.format(time_end-time_start))
        
        optimize = 'True'
        if run_time >= 180:
            optimize = 'False'
        
        row = file[6:].split('/')+ [total_value, total_weight, run_time, optimize]
        writer.writerow(row)

    f.close()
    g.close()