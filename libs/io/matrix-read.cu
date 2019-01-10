#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#define MAXCHAR 1024


/* Returns if character is a valid numeric character i.e. can be found in a string representation of a number
*  '?' is included since it may take the place of an unknown value
*/
bool isNumeric(char a) {
    return (a == '.' || a == '-' || a == '?' || isdigit(a));
}


/* Returns if character is a seperator between two elements
*/
bool isSeperator(char a) {
    return (a == ' ' || a == ',');
}


/* Returns if character is any character in valid representation of data
*/
bool isValid(char a) {
    return (isSeperator(a) || isNumeric(a) || a == '\0' || a == '\n');
}


/* Removes all invalid character from given string and returns number of valid elements
*/
int cleanString(char* str) {
    int j = 0;
    int count = 0;

    for(int i = 0; i < MAXCHAR; i++) {
        if(str[i] == '\0') {
            break;
        }

        // remove all sets of invalid characters with 
        if(!isValid(str[i])) {
            int l;
            for(l = i; l < MAXCHAR && str[l] != '\0' && str[l] != '\n'; l++) {
                if(isSeperator(str[l])) {
                    break;
                }
            }
            i = l-1;

        // remove all invalid characters and valid characters surrounded by
        // invalid characters from string in place.
        } else if(isValid(str[i]) && (i == 0 || isValid(str[i-1])) && ((i == MAXCHAR || str[i] == '\0') || isValid(str[i+1]))) {
            str[j] = str[i];
            j++;
        }
    }
    if (!isNumeric(str[j]) || str[j] == '-') {
        str[j] = '\0';
    }
    j = 0;

    // remove all non-numeric and non-seperator characters and condense seperators
    for(int i = 0; i < MAXCHAR; i++) {

        // if numeric character, move to end of cleaned string
        if(isNumeric(str[i])) {
            if(count == 0) count = 1;
            str[j] = str[i];
            j++;

        // if seperator character, add comma to end of cleaned string
        // and ignore all consecutive seperators
        } else if(isSeperator(str[i])) {
            if(count != 0) count++;
            int l;
            for(l = i; l < MAXCHAR && str[l] != '\0' && isSeperator(str[l]); l++);
            i = l-1;
            str[j] = ',';
            j++;
        } else if(str[i] == '?') {
            str[i] = '0';
        } else if(str[i] == '\0') {
            break;
        }
    }
    
    // ensure string is null terminated
    str[j] = '\0';

    //remove leading comma if exists
    if(str[0] == ',') {
        for(int i = 0; i < MAXCHAR-1; i++) {
            str[i] = str[i+1];
        }
        count--;
    }

    return count;
}


/* Return the number of lines in the file
*  Line count is determined by number of new-line characters,
*   thus dataset files may need to end in empty line for proper function
*/
int getFileLineCount(char* path) {                                 
    FILE* fp = fopen(path, "r");
    if (fp == NULL){
        fprintf(stderr, "\n:::: Could not open file %s\n", path);
        exit(1);
    }

    char ch = 0;
    int lines = 0;

    if (fp == NULL) {
        return 0;
    }

    while(!feof(fp)) {
        ch = fgetc(fp);
        if(ch == '\n') {
            lines++;
        }
    }
    fclose(fp);
    return lines;
}


/* Load matrix data from file into flattened array
*  Dataset files may need to end in empty line for proper function
*/
void readMatrix(char* path, float** matrix, int* m, int* n) {
    int line_read_error_count = 0;
    int current_line = 0;
   
    *n = 0;
    *m = 0;

    int elements = 0;

    FILE* fp;
    char str[MAXCHAR];
    *m = getFileLineCount(path);
    
    *matrix = NULL;

    fp = fopen(path, "r");
    if (fp == NULL){
        fprintf(stderr, "\n:::: Could not open file %s\n", path);
        exit(1);
    }
    while (fgets(str, MAXCHAR, fp) != NULL) {

        int values_count = cleanString(str);

        // if string is empty or too short to contain anything meaningful
        // ignore it and decrement line count by 1
        if(strlen(str) < 2) {
            (*m)--;
        } else {

            // ensure all lines have equal number of values
            if(values_count != *n && *n != 0) {
                fprintf(stderr, "\n:::: Not a valid matrix: unequal row lengths. Line will be skipped.\n");
                fprintf(stderr, "\n:::: ''%s''\n", str);
                line_read_error_count++;
                if(line_read_error_count > 5) {
                    fprintf(stderr, "\n:::: Excess of line read errors: something must be wrong with the file.\n");
                    exit(1);
                }
            } else {
                if(*n == 0) {
                    *n = values_count;
                }
            }

            // ensure matrix has been allocated
            if(*matrix == NULL) {
                (*matrix) = (float*)malloc(*n * *m * sizeof(float));
            }

            // create array of float values from string
            char* str_ptr = str;
            char* num = strtok(str_ptr, ",");
            for(int i = (*n * current_line); i < (*n * (current_line+1)) && num; i++) {

                ((*matrix)[i]) = strtof(num, NULL);
                num = strtok(NULL, ",");
                elements++;
            }

            // line number of current line
            current_line++;
        }
    }

    if (fp) {
        fclose(fp);
    }
}