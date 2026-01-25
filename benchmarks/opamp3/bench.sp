* 3-stage opamp benchmark with DC servo and AC open-loop measurement
.include "models/ptm_180.txt"
.include "opamp3/dut.sp"
VDD vdd 0 1.8
Vcm vcm 0 0.9
Vinp vinp vcm dc=0 ac=1
Vinn vinn vcm dc=0 ac=0
XU1 vinp vinn vout vdd 0 opamp3 w_in=3u l_in=180n w_load=3u l_load=180n w_stage2=4u l_stage2=180n w_stage3=8u l_stage3=180n cc=1p rc=1k ibias=100u
Lservo vinn vout 1e9
CL vout 0 2p
RL vout 0 100k
.end
