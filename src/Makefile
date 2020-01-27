# Optimizations
OPT1=-g -Wall
libs=-lmpi -lm

DIST_NAME=mpi_barcelona_mertens
PDF_NAME=barcelona_mertens_report.pdf
SOURCE_NAME= ../src ../data ../sbatch 

all: lapFusion$(version).c
ifeq ($(version), 00)
	gcc -o lap $(OPT1) lapFusion$(version).c -lm
else ifeq ($(version), 03)
	mpicc -o lap $(OPT1) lapFusion$(version).c -fopenmp $(libs)
else ifeq ($(version), 04)
	mpicc -o lap $(OPT1) lapFusion$(version).c -fopenmp $(libs)
else
	mpicc -o lap lapFusion$(version).c $(libs)
endif

# Clean
clean: 
	rm -rf *.o
realclean: clean
	rm -rf lap

# Generating the zip
send:
	@echo "Preaparing for submission"
	rm -f ../$(DIST_NAME).zip
	mkdir -p ../$(DIST_NAME)
	cp -r $(SOURCE_NAME) ../$(PDF_NAME) ../$(DIST_NAME)/
	zip ../$(DIST_NAME).zip ../$(DIST_NAME)/*
	rm -rf ../$(DIST_NAME)
	@echo "Submission is ready"
