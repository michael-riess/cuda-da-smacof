# DA CUDA SMACOF

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