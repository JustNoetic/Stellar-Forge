import rebound
sim = rebound.Simulation()
sim.add(m=1)
sim.add(m=0.1, a=1)
p = sim.particles[1]
print(f"x={p.x}, y={p.y}, z={p.z}")
