import array
import random

from deap import algorithms, base, creator, tools

from stream.opt.allocation.genetic_algorithm.statistics_evaluator import StatisticsEvaluator


class GeneticAlgorithm:
    def __init__(
        self,
        fitness_evaluator,
        individual_length,
        valid_allocations,
        num_generations=250,
        num_individuals=64,
        pop=None,
    ) -> None:
        if pop is None:
            pop = []
        self.num_generations = num_generations  # number of generations
        self.num_individuals = num_individuals  # number of individuals in initial generation
        self.para_mu = int(num_individuals / 2)  # number of indiviuals taken from previous generation
        self.para_lambda = num_individuals  # number of indiviuals in generation
        self.prob_crossover = 0.3  # probablility to perform corssover
        self.prob_mutation = 0.7  # probablility to perform mutation
        self.valid_allocations = valid_allocations

        self.individual_length = individual_length

        self.fitness_evaluator = fitness_evaluator  # class to evaluate fitness of each indiviual
        # class to track statistics of certain generations
        self.statistics_evaluator = StatisticsEvaluator(self.fitness_evaluator)

        # define target of fitness function
        creator.create("FitnessMulti", base.Fitness, weights=self.fitness_evaluator.weights)
        # define individual in population
        creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMulti)  # type: ignore

        self.toolbox = base.Toolbox()  # initialize DEAP toolbox
        self.hof = tools.ParetoFront()  # initialize Hall-of-Fame as Pareto Front

        def get_random_individual():
            """rReturns a random individual by randomly choosing from the valid allocations of each node"""
            return (random.choice(x) for x in valid_allocations)

        # attribute generator
        self.toolbox.register(
            "attr_bool", get_random_individual
        )  # single attribute of indiviuals can encode core allocation for HW

        # structure initializers
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,  # type: ignore
            self.toolbox.attr_bool,  # type: ignore
        )  # indivual has #nodes in graph attributes
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,  # type: ignore
        )  # define polulation based on indiviudal

        # link user defined fitness function to toolbox
        self.toolbox.register("evaluate", self.fitness_evaluator.get_fitness)

        individual_length_threshold = 10
        if self.individual_length > individual_length_threshold:
            self.toolbox.register("mate", tools.cxOrdered)  # for big graphs use cxOrdered crossover function
        else:
            self.toolbox.register("mate", tools.cxTwoPoint)  # for small graphs use two point crossover function

        # link user defined mutation function to toolbox
        self.toolbox.register("mutate", self.mutate)
        # use non-dominated sorting genetic algorithm for multi-objective optimization
        self.toolbox.register("select", tools.selNSGA2)

        # populate random initial generation
        self.pop = self.toolbox.population(n=self.num_individuals)  # type: ignore

        # replace sub part of initial generation with user provided individuals
        for indv_index in range(len(pop)):
            for i in range(self.fitness_evaluator.workload.number_of_nodes()):
                self.pop[indv_index][i] = pop[indv_index][i]

            # don't bias initial population too much
            if indv_index >= self.num_individuals / 4:
                break

    def run(self):
        # plot statistics during evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register(
            "avg (" + ", ".join(self.fitness_evaluator.metrics) + ")",
            self.statistics_evaluator.get_avg,
        )
        stats.register(
            "std (" + ", ".join(self.fitness_evaluator.metrics) + ")",
            self.statistics_evaluator.get_std,
        )
        stats.register(
            "min (" + ", ".join(self.fitness_evaluator.metrics) + ")",
            self.statistics_evaluator.get_min,
        )
        stats.register(
            "max (" + ", ".join(self.fitness_evaluator.metrics) + ")",
            self.statistics_evaluator.get_max,
        )
        # stats.register("saved", self.save_population)

        algorithms.eaMuPlusLambda(
            self.pop,
            self.toolbox,
            mu=self.para_mu,
            lambda_=self.para_lambda,
            cxpb=self.prob_crossover,
            mutpb=self.prob_mutation,
            ngen=self.num_generations,
            stats=stats,
            halloffame=self.hof,
        )
        return self.pop, self.hof

    def mutate(self, individual):
        prob_mutation = 1 / len(individual)

        # change one of the position's core allocation
        change_percentage = 0.75
        if random.random() < change_percentage:
            for position in range(len(list(individual))):
                individual[position]
                if random.random() < prob_mutation:
                    current_core_allocation = individual[position]
                    valid_new_core_allocations = sorted(
                        set(self.valid_allocations[position]) - set([current_core_allocation])
                    )
                    individual[position] = random.choice(valid_new_core_allocations)
        # swap the core allocation of two randomly chosen positions
        else:
            first_position, second_position = random.sample(range(len(individual)), 2)
            tmp = individual[second_position]
            individual[second_position] = individual[first_position]
            individual[first_position] = tmp

        return (individual,)

    def save_population(self, x):
        if self.statistics_evaluator.current_generation % self.statistics_evaluator.evaluation_periode == 0:
            self.statistics_evaluator.append_generation(list(self.pop))
            self.statistics_evaluator.current_generation += 1
            return True
        else:
            self.statistics_evaluator.current_generation += 1
            return False
