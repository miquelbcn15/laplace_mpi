size=512
for i = 4; i <= 128; i*2 
mpirun np $i ./lap $size 100
size=size*2
