
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
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
    

    def plot_evolution(self):
        first_values = []
        second_values = []
        third_values = []

        for pop in self.generations:
            y = []
            x = []
            z = []
            third_metirc = 0

            for individual in pop:
                if len(self.fitness_evaluator.metrics) == 1:
                    first_metric = second_metric = self.fitness_evaluator.get_fitness(individual)
                elif len(self.fitness_evaluator.metrics) == 2:
                    first_metric, second_metric = self.fitness_evaluator.get_fitness(individual)
                elif len(self.fitness_evaluator.metrics) == 3:
                    first_metric, second_metric, third_metirc = self.fitness_evaluator.get_fitness(individual)

                y.append(second_metric)
                x.append(first_metric)
                z.append(third_metirc)

            first_values.append(x)
            second_values.append(y)
            if len(self.fitness_evaluator.metrics) == 3:
                third_values.append(z)


        plt.clf()
        for data_row in range(len(first_values)):
            
            if data_row != len(first_values)-1:
                value = 1/(data_row+1.1)
                col = (value, value, value)
            else:
                col = (1, 0, 0)
            
            plt.scatter(first_values[data_row], second_values[data_row], color=col, label=self.num_generation[data_row])
        
        plt.title("Evolution of solutions in population over time")
        plt.grid()
        plt.xlabel(self.fitness_evaluator.metrics[0])
        if len(self.fitness_evaluator.metrics) == 1:
            plt.ylabel(self.fitness_evaluator.metrics[0])
        else:
            plt.ylabel(self.fitness_evaluator.metrics[1])
        # plt.legend(loc='upper right')
        plt.show() 

        if len(self.fitness_evaluator.metrics) == 3:
            plt.clf()
            for data_row in range(len(first_values)):
                
                if data_row != len(first_values)-1:
                    value = 1/(data_row+1.1)
                    col = (value, value, value)
                else:
                    col = (1, 0, 0)
                
                plt.scatter(first_values[data_row], third_values[data_row], color=col, label=self.num_generation[data_row])
            
            plt.title("Evolution of solutions in population over time")
            plt.grid()
            plt.xlabel(self.fitness_evaluator.metrics[0])
            plt.ylabel(self.fitness_evaluator.metrics[2])
            # plt.legend(loc='upper right')
            plt.show()      


    def plot_population(self, pop):
        # funcion only works if more than two fitness evaluator metrics are used

        y = []
        x = []
        z = []

        for individual in pop:
            if len(self.fitness_evaluator.metrics) == 2:
                first_metric, second_metric = self.fitness_evaluator.get_fitness(individual)
                third_metirc = first_metric # unrelevant assignment
            elif len(self.fitness_evaluator.metrics) == 3:
                first_metric, second_metric, third_metirc = self.fitness_evaluator.get_fitness(individual)

            x.append(first_metric)
            y.append(second_metric)
            z.append(third_metirc)

        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        min_z = min(z)
        max_z = max(z)

        min_max_y = '['+ str(numpy.format_float_scientific(min_y, precision=2)) + ', ' + str(numpy.format_float_scientific(max_y, precision=2)) + ']'
        min_max_x = '['+ str(numpy.format_float_scientific(min_x, precision=2)) + ', ' + str(numpy.format_float_scientific(max_x, precision=2)) + ']'
        min_max_z = '['+ str(numpy.format_float_scientific(min_z, precision=2)) + ', ' + str(numpy.format_float_scientific(max_z, precision=2)) + ']'
    
        if len(self.fitness_evaluator.metrics) == 2:
            plt.scatter(x, y, c ="blue")
            plt.title("Pareto front of produced solutions")
            plt.grid()
            plt.xlabel(self.fitness_evaluator.metrics[0] + "\n" + min_max_x)
            if len(self.fitness_evaluator.metrics) == 1:
                plt.ylabel(self.fitness_evaluator.metrics[0] + "\n" + min_max_y)
            else:
                plt.ylabel(self.fitness_evaluator.metrics[1] + "\n" + min_max_y)
            plt.tight_layout()
            plt.show()

        else:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            import numpy as np
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(y, x, z, c=z)
            ax.set_xlabel(self.fitness_evaluator.metrics[1])
            ax.set_ylabel(self.fitness_evaluator.metrics[0])
            ax.set_zlabel(self.fitness_evaluator.metrics[2])
            ax.locator_params(axis='x', nbins=6)
            ax.locator_params(axis='y', nbins=6)
            ax.view_init(15, 45)
            path = "outputs/paretofront_middle"
            plt.savefig(path)
            # plt.show()

    def print_population(self, pop):
        for individual in pop:
            print(list(individual), round(self.fitness_evaluator.get_energy(individual), 2), round(self.fitness_evaluator.get_execution_time(individual), 2))

    
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
