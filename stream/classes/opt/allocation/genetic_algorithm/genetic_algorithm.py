import itertools
import logging

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import array
import random

from stream.classes.opt.allocation.genetic_algorithm.statistics_evaluator import StatisticsEvaluator

class GeneticAlgorithm:
    def __init__(self, fitness_evaluator, individual_length, valid_allocations, num_generations=250, num_individuals=64, pop=[]) -> None:
        self.num_generations = num_generations  # number of generations
        self.num_individuals = num_individuals  # number of individuals in initial generation
        self.para_mu = int(num_individuals/2)   # number of indiviuals taken from previous generation
        self.para_lambda = num_individuals      # number of indiviuals in generation
        self.prob_crossover = 0.3               # probablility to perform corssover
        self.prob_mutation = 0.7                # probablility to perform mutation
        self.valid_allocations = valid_allocations

        self.individual_length = individual_length

        self.fitness_evaluator = fitness_evaluator                              # class to evaluate fitness of each indiviual
        self.statistics_evaluator = StatisticsEvaluator(self.fitness_evaluator) # class to track statistics of certain generations

        creator.create("FitnessMulti", base.Fitness, weights=self.fitness_evaluator.weights)  # define target of fitness function
        creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMulti) # define individual in population

        self.toolbox = base.Toolbox()   # initialize DEAP toolbox
        self.hof = tools.ParetoFront()  # initialize Hall-of-Fame as Pareto Front

        # Create a function that returns a random individual by randomly choosing from the valid allocations of each node
        def get_random_individual():
            return (random.choice(l) for l in valid_allocations)


        # attribute generator
        self.toolbox.register("attr_bool", get_random_individual)   # single attribute of indiviuals can encode core allocation for HW

        # structure initializers
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_bool)  # indivual has #nodes in graph attributes
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)  # define polulation based on indiviudal

        # link user defined fitness function to toolbox
        self.toolbox.register("evaluate", self.fitness_evaluator.get_fitness)

        if self.individual_length > 10:
            self.toolbox.register("mate", tools.cxOrdered)  # for big graphs use cxOrdered crossover function
        else:
            self.toolbox.register("mate", tools.cxTwoPoint) # for small graphs use two point crossover function

        # link user defined mutation function to toolbox 
        self.toolbox.register("mutate", self.mutate)
        # use non-dominated sorting genetic algorithm for multi-objective optimization
        self.toolbox.register("select", tools.selNSGA2)

        # populate random initial generation
        self.pop = self.toolbox.population(n=self.num_individuals)              

        # replace sub part of initial generation with user provided individuals
        for indv_index in range(len(pop)):
            for i in range(self.fitness_evaluator.workload.number_of_nodes()):
                self.pop[indv_index][i] = pop[indv_index][i]
            
            # don't bias initial population too much
            if indv_index >= self.num_individuals/4:
                break

    def run(self):
        # plot statistics during evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg (" + ', '.join(self.fitness_evaluator.metrics) + ")", self.statistics_evaluator.get_avg)
        stats.register("std (" + ', '.join(self.fitness_evaluator.metrics) + ")", self.statistics_evaluator.get_std)
        stats.register("min (" + ', '.join(self.fitness_evaluator.metrics) + ")", self.statistics_evaluator.get_min)
        stats.register("max (" + ', '.join(self.fitness_evaluator.metrics) + ")", self.statistics_evaluator.get_max)
        # stats.register("saved", self.save_population)

        logbook = algorithms.eaMuPlusLambda(self.pop, self.toolbox, mu=self.para_mu, lambda_=self.para_lambda, cxpb=self.prob_crossover, 
                                    mutpb=self.prob_mutation, ngen=self.num_generations, stats=stats, halloffame=self.hof)

        # latency = []
        # buffer_size = []
        # energy = []

        # if len(self.fitness_evaluator.metrics) == 3:
        #     for index in range(len(logbook[1])):
        #         latency.append(logbook[1][index]['min (latency, buffer_size, energy)'][0])
        #         buffer_size.append(logbook[1][index]['min (latency, buffer_size, energy)'][1])
        #         energy.append(logbook[1][index]['min (latency, buffer_size, energy)'][2])

        # file1 = open("outputs/logbook.py","w+")
        # file1.write("latency = " + str(latency) + "\n")
        # file1.write("buffer_size = " + str(buffer_size) + "\n")
        # file1.write("energy = " + str(energy) + "\n")
        # file1.close()

        # file1 = open("outputs/log.txt","a")
        # file1.write(str(logbook[1]))
        # file1.close()

        # if len(self.fitness_evaluator.metrics) > 1:
            # self.statistics_evaluator.plot_population(self.hof)

        return self.pop, self.hof

    def mutate(self, individual):
        prob_mutation = 1/len(individual)
        
        # change one of the position's core allocation
        if random.random() < 0.75:
            for position in range(len(list(individual))):
                old_value = individual[position]
                if random.random() < prob_mutation:
                    current_core_allocation = individual[position]
                    valid_new_core_allocations = sorted(set(self.valid_allocations[position]) - set([current_core_allocation]))
                    individual[position] = random.choice(valid_new_core_allocations)
        # swap the core allocation of two randomly chosen positions
        else:
            first_position, second_position = random.sample(range(len(individual)), 2)
            tmp = individual[second_position]
            individual[second_position] = individual[first_position]
            individual[first_position] = tmp

        return individual,

    def save_population(self, x):
        if (self.statistics_evaluator.current_generation % self.statistics_evaluator.evaluation_periode == 0):
            self.statistics_evaluator.append_generation(list(self.pop))
            self.statistics_evaluator.current_generation += 1
            return True
        else:
            self.statistics_evaluator.current_generation += 1
            return False