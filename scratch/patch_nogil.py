def main():
    with open('d:/Files/Coding/OpenGL/Stellar-Forge/physics_core.py', 'r') as f:
        content = f.read()

    # Replace all @njit(cache=True) with @njit(cache=True, nogil=True)
    content = content.replace("@njit(cache=True)", "@njit(cache=True, nogil=True)")
    
    # Also just in case there are any naked @njit
    content = content.replace("@njit\n", "@njit(nogil=True)\n")

    with open('d:/Files/Coding/OpenGL/Stellar-Forge/physics_core.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    main()
