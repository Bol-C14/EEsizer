* Simple OTA core
.subckt ota vinp vinn vout vdd vss w_in=4u l_in=180n w_load=4u l_load=180n cc=1p ibias=50u
I1 tail vss {ibias}
M1 n1 vinp tail vss nmos W={w_in} L={l_in}
M2 vout vinn tail vss nmos W={w_in} L={l_in}
M3 n1 n1 vdd vdd pmos W={w_load} L={l_load}
M4 vout n1 vdd vdd pmos W={w_load} L={l_load}
Cc vout n1 {cc}
.ends ota
