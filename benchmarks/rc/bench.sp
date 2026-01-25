* RC lowpass benchmark
.include "models/ptm_180.txt"
VDD vdd 0 1.8
Vin vin 0 dc=0 ac=1
R1 vin vout 1k
C1 vout 0 1n
.end
