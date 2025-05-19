from force_simulation import TearSimulator

sim = TearSimulator(stiffness=4.5, tear_threshold=20)

print("=== Tearing Simulation Test ===")

for d in [1, 2, 3, 4, 5, 6]:
    f, torn, progress = sim.apply_displacement(d)
    print(f"Displacement: {d} mm → Force: {f:.2f} N | Tear: {torn} | Progress: {progress:.1f}%")

print("\nResetting...")
sim.reset()
print(sim.__dict__)
