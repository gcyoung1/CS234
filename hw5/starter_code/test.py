from sim import Simulator
import random

for update_strategy in ['none', 'popular', 'corrective', 'counterfactual']:
    print(f"\n\nTesting {update_strategy}")
    sim = Simulator(num_users=3, num_arms=3, num_features=3, update_freq=5, update_arms_strategy=update_strategy)
    for step in range(25):
        user_id = random.randint(0, sim.num_users - 1)
        arm_id = random.randint(0, sim.num_arms - 1)
        sim.step(user_id, arm_id)
        
print(f"Added arm:\n {self.arms[self.num_arms-1]}")

