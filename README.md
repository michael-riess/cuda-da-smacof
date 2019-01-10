# DA CUDA SMACOF

## Arguments
### SMACOF
1. matrix-file: string/text
2. blocks: integer
3. threads: integer (per block)
4. s: integer (lower dimensional target)
5. epsilong: decimal/float (lower error bound)
6. k-max: integer (maximum number of iterations within SMACOF)
7. iterations: integer (number of times da-smacof will be performed (to find averages and medians))
8. track-median: "median" (if text "median" is used, median results will be recorded)

### DA-SMACOF
1. matrix-file: string/text
2. blocks: integer
3. threads: integer (per block)
4. s: integer (lower dimensional target)
5. epsilong: decimal/float (lower error bound)
6. k-max: integer (maximum number of iterations within SMACOF)
7. temp-min: decimal/float (minimum temperature before running final SMACOF round without annealing)
8. alpha: decimal/float (temperature reduction factor)
9. iterations: integer (number of times da-smacof will be performed (to find averages and medians))
10. track-median: "median" (if text "median" is used, median results will be recorded)

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
To compile just the smacof program
```
make smacof
```
To compile smacof for debugging
```
make memcheck-smacof
```
To compile just the da-smacof program
```
make da-smacof
```
To compile da-smacof for debugging
```
make memcheck-da-smacof
```