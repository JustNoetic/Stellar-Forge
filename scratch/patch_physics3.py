def main():
    with open('d:/Files/Coding/OpenGL/Stellar-Forge/physics_core.py', 'r') as f:
        content = f.read()

    # 1. Replace sim = rebound.Simulation()
    content = content.replace("sim = rebound.Simulation()", "sim = Simulation()")
    
    # 2. Remove sim.softening and sim.integrator
    content = content.replace("sim.softening = 1e-6\n    sim.integrator = \"ias15\"\n", "")
    
    # 3. Replace rebound.ParticleNotFound with KeyError
    content = content.replace("except rebound.ParticleNotFound:", "except KeyError:")
    
    with open('d:/Files/Coding/OpenGL/Stellar-Forge/physics_core.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    main()
