* OTA benchmark with DC servo and AC open-loop measurement
.include "models/ptm_180.txt"
.include "ota/dut.sp"
VDD vdd 0 1.8
Vcm vcm 0 0.9
Vinp vinp vcm dc=0 ac=1
Vinn vinn vcm dc=0 ac=0
XU1 vinp vinn vout vdd 0 ota w_in=4u l_in=180n w_load=4u l_load=180n cc=1p ibias=50u
Lservo vinn vout 1e9
CL vout 0 1p
.end
