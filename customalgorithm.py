#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generally, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random

from deap import tools

def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    """This is algorithm implements the ask-tell model proposed in
    [Colette2010]_, where ask is called `generate` and tell is called `update`.

    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm generates the individuals using the :func:`toolbox.generate`
    function and updates the generation method with the :func:`toolbox.update`
    function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The pseudocode goes as follow ::

        for g in range(ngen):
            population = toolbox.generate()
            evaluate(population)
            toolbox.update(population)


    This function expects :meth:`toolbox.generate` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    
    .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
       R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
       Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
       Multidisciplinary Design Optimization in Computational Mechanics,
       Wiley, pp. 527-565;

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    covs = []
    sigmas = []
    centroids = []

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Update the strategy with the evaluated individuals
        C, sigma, centroid = toolbox.update(population)
        covs.append(C), sigmas.append(sigma), centroids.append(centroid)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        #if verbose:
        #    print logbook.stream

    return population, logbook, covs, sigmas, centroids
