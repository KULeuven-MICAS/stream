import numpy


class StatisticsEvaluator:
    def __init__(self, fitness_evaluator) -> None:
        self.fitness_evaluator = fitness_evaluator

        self.generations = []
        self.num_generation = []
        self.current_generation = 0
        self.evaluation_periode = 10

    def append_generation(self, pop):
        self.generations.append(list(pop))
        self.num_generation.append(self.current_generation)

    def print_population(self, pop):
        for individual in pop:
            print(
                list(individual),
                round(self.fitness_evaluator.get_energy(individual), 2),
                round(self.fitness_evaluator.get_execution_time(individual), 2),
            )

    def get_avg(self, x):
        means = numpy.mean(x, axis=0).tolist()

        for i in range(len(means)):
            means[i] = numpy.format_float_scientific(means[i], precision=2, min_digits=2)

        return means

    def get_std(self, x):
        stds = numpy.std(x, axis=0).tolist()

        for i in range(len(stds)):
            stds[i] = numpy.format_float_scientific(stds[i], precision=2, min_digits=2)

        return stds

    def get_min(self, x):
        mins = numpy.min(x, axis=0).tolist()

        for i in range(len(mins)):
            mins[i] = numpy.format_float_scientific(mins[i], precision=2, min_digits=2)

        return mins

    def get_max(self, x):
        maxs = numpy.max(x, axis=0).tolist()

        for i in range(len(maxs)):
            maxs[i] = numpy.format_float_scientific(maxs[i], precision=2, min_digits=2)

        return maxs
