import os

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    content = content.replace("import rebound", "from physics_core import Simulation")
    content = content.replace("sim = rebound.Simulation()", "sim = Simulation()")
    content = content.replace("sim.units = ('AU', 'yr', 'Msun')", "sim.G = 39.476926421373")
    content = content.replace("sim.integrator = \"ias15\"", "")
    
    with open(filepath, 'w') as f:
        f.write(content)

def main():
    patch_file('d:/Files/Coding/OpenGL/Stellar-Forge/scripts/accuracy_test.py')
    patch_file('d:/Files/Coding/OpenGL/Stellar-Forge/scripts/accuracy_test_no_gr.py')

if __name__ == '__main__':
    main()
