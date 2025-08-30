#!/usr/bin/env python3
"""
Hallo Welt Programm mit numpy Test
"""
import numpy as np

def main():
    print("Hallo Welt!")
    print("Willkommen zu deinem Python-Projekt!")
    
    # Numpy Test
    print("\n--- Numpy Test ---")
    
    # Erstelle ein einfaches Array
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Array: {arr}")
    print(f"Summe: {np.sum(arr)}")
    print(f"Durchschnitt: {np.mean(arr)}")
    
    # Erstelle eine 2D Matrix
    matrix = np.array([[1, 2], [3, 4]])
    print(f"Matrix:\n{matrix}")
    print(f"Matrix Determinante: {np.linalg.det(matrix)}")
    
    print(f"Numpy Version: {np.__version__}")

if __name__ == "__main__":
    main()