
import mesa
from agents import GrassPatch, Sheep, Wolf

from typing import Type, Optional, Callable

import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
from addons.visuals import plot_avg_std, plot_experiment
# import numpy as np
# import pandas as pd


class RandomActivationByTypeFiltered(mesa.time.RandomActivationByType):
    """
    A scheduler that overrides the get_type_count method to allow for filtering
    of agents by a function before counting.

    Example:
    >>> scheduler = RandomActivationByTypeFiltered(model)
    >>> scheduler.get_type_count(AgentA, lambda agent: agent.some_attribute > 10)
    """

    def get_type_count(
        self,
        type_class: Type[mesa.Agent],
        filter_func: Optional[Callable[[mesa.Agent], bool]] = None,
    ) -> int:
        """
        Returns the current number of agents of certain type in the queue
        that satisfy the filter function.
        """
        if type_class not in self.agents_by_type:
            return 0
        count = 0
        for agent in self.agents_by_type[type_class].values():
            if filter_func is None or filter_func(agent):
                count += 1
        return count


class WolfSheep(mesa.Model):
    """
    Wolf-Sheep Predation Model
    """

    height = 20
    width = 20

    initial_sheep = 100
    initial_wolves = 50

    sheep_reproduce = 0.04
    wolf_reproduce = 0.05

    wolf_gain_from_food = 20

    grass = False
    grass_regrowth_time = 30
    sheep_gain_from_food = 4

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        width=20,
        height=20,
        initial_sheep=100,
        initial_wolves=50,
        sheep_reproduce=0.04,
        wolf_reproduce=0.05,
        wolf_gain_from_food=20,
        grass=True,
        grass_regrowth_time=30,
        sheep_gain_from_food=4, 
    ):
        """
        Create a new Wolf-Sheep model with the given parameters.

        Args:
            initial_sheep: Number of sheep to start with
            initial_wolves: Number of wolves to start with
            sheep_reproduce: Probability of each sheep reproducing each step
            wolf_reproduce: Probability of each wolf reproducing each step
            wolf_gain_from_food: Energy a wolf gains from eating a sheep
            grass: Whether to have the sheep eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                 once it is eaten
            sheep_gain_from_food: Energy sheep gain from grass, if enabled.
        """
        super().__init__()
        # Set parameters
        self.width = width
        self.height = height
        self.initial_sheep = initial_sheep
        self.initial_wolves = initial_wolves
        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.wolf_gain_from_food = wolf_gain_from_food
        self.grass = grass
        self.grass_regrowth_time = grass_regrowth_time
        self.sheep_gain_from_food = sheep_gain_from_food

        #self.schedule = mesa.time.BaseScheduler(self)
        self.schedule = RandomActivationByTypeFiltered(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self.datacollector = mesa.DataCollector(
            {
                "Wolves": lambda m: m.schedule.get_type_count(Wolf),
                "Sheep": lambda m: m.schedule.get_type_count(Sheep),
                "Grass": lambda m: m.schedule.get_type_count(
                    GrassPatch, lambda x: x.fully_grown
                ),
            }
        )

        # Create sheep:
        for _ in range(self.initial_sheep):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = 2 * self.sheep_gain_from_food          #self.random.randrange(2 * self.sheep_gain_from_food)
            sheep = Sheep(self.next_id(), (x, y), self, True, energy)
            self.grid.place_agent(sheep, (x, y))
            self.schedule.add(sheep)

        # Create wolves
        for _ in range(self.initial_wolves):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            #energy = self.random.randrange(2 * self.wolf_gain_from_food)
            energy = 2 * self.wolf_gain_from_food
            wolf = Wolf(self.next_id(), (x, y), self, True, energy)
            self.grid.place_agent(wolf, (x, y))
            self.schedule.add(wolf)

        # Create grass patches
        if self.grass:
            for _, (x, y) in self.grid.coord_iter():
                fully_grown = self.random.choice([True, False])

                if fully_grown:
                    countdown = self.grass_regrowth_time
                else:
                    countdown = self.random.randrange(self.grass_regrowth_time)

                patch = GrassPatch(self.next_id(), (x, y), self, fully_grown, countdown)
                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()
        self.datacollector.collect(self)
        #print(self.datacollector.get_model_vars_dataframe())


if __name__ == "__main__":
    # wolfs = [10, 15, 20]
    # sheeps = [10, 15, 20]
    wolfs = [10]
    sheeps = [50]
    seeds = [7123, 1287, 6372, 2651, 199]

    for wolf in wolfs:
        for sheep in sheeps:
            for seed in seeds:
                args = {
                    "width": 20,
                    "height": 20,
                    "initial_sheep": sheep,
                    "initial_wolves": wolf,
                    "sheep_reproduce": 0.04,
                    "wolf_reproduce": 0.2,
                    "wolf_gain_from_food": 20,
                    "grass": True,
                    "grass_regrowth_time": 30,
                    "sheep_gain_from_food": 4, 
                }
                random.seed(seed)

                model = WolfSheep(**args)
                model.run_model()
                data = model.datacollector.get_model_vars_dataframe()
                data.to_csv(f"./exp/experiment_w_{wolf}_s_{sheep}_seed_{seed}")
                # wandb.init(
                #     project="wolf_sheep", 
                #     name=f"experiment_w_{wolf}_s_{sheep}_seed_{seed}", 
                #     config={
                #     "seed": seed,
                # })

                # for index, row in data.iterrows():
                #     wandb.log({'wolves': row['Wolves']})
                #     wandb.log({'sheep': row['Sheep'] })
                #     wandb.log({'grass': row['Grass'] })

                # wandb.finish()
            #paths = ["./exp/experiment_w_10_s_10_seed_7123", "./exp/experiment_w_10_s_10_seed_1287", "./exp/experiment_w_10_s_10_seed_6372", "./exp/experiment_w_10_s_10_seed_2651", "./exp/experiment_w_10_s_10_seed_199"]
            paths = [f"./exp/experiment_w_{wolf}_s_{sheep}_seed_{seed}" for seed in seeds]
            plot_experiment(paths, f"./resources/w_{wolf}_s_{sheep}.png")
            plot_avg_std(paths, f"./resources/w_{wolf}_s_{sheep}_avg.png")
            #fig, axes = plt.subplots(3, 1, figsize=(10, 10))
            #for path in paths:
            #    df = pd.read_csv(path)
            #    sns.lineplot(df['Wolves'], ax=axes[0])
            #    sns.lineplot(df['Sheep'], ax=axes[1])
            #    sns.lineplot(df['Grass'], ax=axes[2])

            #plt.savefig(f"./resources/w_{wolf}_s_{sheep}.png")

