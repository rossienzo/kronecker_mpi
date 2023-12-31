from mpi4py import MPI
import numpy as np
import pandas as pd

ROWS = 4
COLS = 10

def gen_matrix(rows, cols):
    matrix = []

    for _ in range(rows):
        matrix.append([np.random.random() for _ in range(cols)])
    return matrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Duas matrizes de exemplo
matrix_A = np.array(gen_matrix(ROWS, COLS))
matrix_B = np.array(gen_matrix(ROWS, COLS))

# Tamanho das matrizes
rows_A, cols_A = len(matrix_A), len(matrix_A[0])
rows_B, cols_B = len(matrix_B), len(matrix_B[0])


# Tamanho das matrizes resultantes
result_rows = rows_A * rows_B
result_cols = cols_A * cols_B

# Divide o trabalho entre os processos
local_rows = result_rows // size
local_cols = result_cols

# Crie matrizes locais para armazenar as partes locais do resultado
local_result = np.zeros((local_rows, local_cols), dtype=float)

# Distribua partes da matriz_A para todos os processos
local_A = np.empty((local_rows, cols_A), dtype=float)
comm.Scatter(matrix_A, local_A, root=0)

# Calcula a parte local do produto de Kronecker
for i in range(local_rows):
    global_row = rank * local_rows + i
    for j in range(local_cols):
        global_col = j
        local_result[i, j] = local_A[i // rows_B, global_col // cols_B] * matrix_B[global_row % rows_B, global_col % cols_B]

# Colete os resultados locais de todos os processos
if rank == 0:
    result = np.empty((result_rows, result_cols), dtype=local_result.dtype)
else:
    result = None

comm.Gather(local_result, result, root=0)

# O processo 0 imprime a matriz resultante
if rank == 0:
    print("Resultado do Produto de Kronecker:")
    print(result.round(4))