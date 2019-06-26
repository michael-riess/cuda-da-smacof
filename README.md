# CUDA DA SMACOF

## Arguments
### SMACOF
1. matrix-file: string/text
2. s: integer (lower dimensional target)
3. epsilong: decimal/float (lower error bound)
4. k-max: integer (maximum number of iterations within SMACOF)
5. iterations: integer (number of times da-smacof will be performed (to find averages and medians))
6. 
    - track-median: if `median` is used, the time and stress of median results will be recorded.
    - track-median-solution: if `median_solution` is used, the median mapping solution will be recorded.
    - track-median-stresses: if `median_stresses` is used, all internal stress values for the median mapping solution will be recorded.

**example**
`./smacof ./test.data 2 .00001 1000 50 median`

### CUDA-SMACOF
1. matrix-file: string/text
2. blocks: integer
3. threads: integer (per block)
4. s: integer (lower dimensional target)
5. epsilong: decimal/float (lower error bound)
6. k-max: integer (maximum number of iterations within SMACOF)
7. iterations: integer (number of times da-smacof will be performed (to find averages and medians))
8.
    - track-median: if `median` is used, the time and stress of median results will be recorded.
    - track-median-solution: if `median_solution` is used, the median mapping solution will be recorded.
    - track-median-stresses: if `median_stresses` is used, all internal stress values for the median mapping solution will be recorded.

**example**
`./da-cuda-smacof ./test.data 512 512 2 .00001 1000 50 median_stresses`

### CUDA-DA-SMACOF
1. matrix-file: string/text
2. blocks: integer
3. threads: integer (per block)
4. s: integer (lower dimensional target)
5. epsilong: decimal/float (lower error bound)
6. k-max: integer (maximum number of iterations within SMACOF)
7. temp-min: decimal/float (minimum temperature before running final SMACOF round without annealing)
8. alpha: decimal/float (temperature reduction factor)
9. iterations: integer (number of times da-smacof will be performed (to find averages and medians))
10. 
    - track-median: if `median` is used, the time and stress of median results will be recorded.
    - track-median-solution: if `median_solution` is used, the median mapping solution will be recorded.
    - track-median-stresses: if `median_stresses` is used, all internal stress values for the median mapping solution will be recorded.

**example**
`./da-cuda-smacof ./test.data 512 512 2 .00001 1000 .1 .90 25 media_solution`


## Matrices
- input matrix files require an empty terminating line for proper functionality
### Supported Matrix File Formats
- CSV / TSV (and most other deliminator spaced value formats)
- Matrix Market Format

## Compiling
For most use cases, you will want to compile everything; to do so, run the following command
```
make all
```
To just compile the associated libraries
```
make libraries
```
To clean compiled libraries
```
make clean
```