import mesa

import seaborn as sns
import numpy as np
import pandas as pd

import random


random.seed(0)


class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1

    # def step(self):
    #     self.wealth += random.randint(1, 10)
    #     print(f"Hi, I am an agent, you can call me {str(self.unique_id)} and {self.wealth}.")

    def step(self):
        # Verify agent has some wealth
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.schedule.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1
        print(f"Hi, I am an agent, you can call me {str(self.unique_id)} and {self.wealth}.")


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        super().__init__()
        self.num_agents = N
        # Create scheduler and assign it to the model
        self.schedule = mesa.time.BaseScheduler(self)
        #self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            # Add the agent to the scheduler
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()


starter_model = MoneyModel(10)
for i in range(4):
    starter_model.step()
    print()