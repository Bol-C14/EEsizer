* Three-stage opamp core
.subckt opamp3 vinp vinn vout vdd vss w_in=3u l_in=180n w_load=3u l_load=180n w_stage2=4u l_stage2=180n w_stage3=8u l_stage3=180n cc=1p rc=1k ibias=100u
I1 tail vss {ibias}
M1 n1 vinp tail vss nmos W={w_in} L={l_in}
M2 n2 vinn tail vss nmos W={w_in} L={l_in}
M3 n1 n1 vdd vdd pmos W={w_load} L={l_load}
M4 n2 n1 vdd vdd pmos W={w_load} L={l_load}
M5 n3 n2 vdd vdd pmos W={w_stage2} L={l_stage2}
M6 n3 n3 vss vss nmos W={w_stage2} L={l_stage2}
M7 vout n3 vdd vdd pmos W={w_stage3} L={l_stage3}
M8 vout n3 vss vss nmos W={w_stage3} L={l_stage3}
Cc n2 vout {cc}
Rc n2 vout {rc}
.ends opamp3
