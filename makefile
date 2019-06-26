#Makefile

#define variables
SMACOF_LIBS= $(wildcard libs/*)
COMPILER= nvcc
OPT=
TEST_OPT= -g -G
LIBS= -lm -lcublas -lcurand
INCLUDES_PATH= includes/


#compile
all:
	$(COMPILER) $(OPT) smacof.cu $(wildcard libraries/*) -I$(INCLUDES_PATH) $(LIBS) -o smacof
	$(COMPILER) $(OPT) cuda-smacof.cu $(wildcard libraries/*) -I$(INCLUDES_PATH) $(LIBS) -o cuda-smacof
	$(COMPILER) $(OPT) da-cuda-smacof.cu $(wildcard libraries/*) -I$(INCLUDES_PATH) $(LIBS) -o da-cuda-smacof
	$(COMPILER) $(TEST-OPT) cuda-smacof.cu $(wildcard libraries/*) -I$(INCLUDES_PATH) $(LIBS) -o memcheck-cuda-smacof
	$(COMPILER) $(TEST-OPT) da-cuda-smacof.cu $(wildcard libraries/*) -I$(INCLUDES_PATH) $(LIBS) -o memcheck-da-cuda-smacof

# compile smacof library files, and move object and header files to libraries/ and includes/ respectively
libraries:
	mkdir libraries/
	mkdir includes/
	for dir in $(SMACOF_LIBS); do \
		cd $$dir; \
		$(COMPILER) -c *.cu $(LIBS); \
		mv *.o ../../libraries; \
		cp *.h ../../includes; \
		cd -; \
	done

#clean Makefile
clean:
	rm -rf libraries/
	rm -rf includes/


#end of Makefile