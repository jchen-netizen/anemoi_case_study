# Useful commands and tips to run Verif
* 6-hour forecast: \
`verif input1.nc input2.nc -m timeseries -o 6 -f Timeseries_6h.png -fs 12,6 -lw 0.25 -ms 2 -tickfs 6`
* 12-hour forecast: \
`verif input1.nc input2.nc -m timeseries -o 12 -f Timeseries_12h.png -fs 12,6 -lw 0.25 -ms 2 -tickfs 6`
* 18-hour forecast: \
`verif input1.nc input2.nc -m timeseries -o 18 -f Timeseries_18h.png -fs 12,6 -lw 0.25 -ms 2 -tickfs 6` 
* 24-hour forecast: \
`verif input1.nc input2.nc -m timeseries -o 24 -f Timeseries_24h.png -fs 12,6 -lw 0.25 -ms 2 -tickfs 6`
* 36-hour forecast: \
`verif input1.nc input2.nc -m timeseries -o 36 -f Timeseries_36h.png -fs 12,6 -lw 0.25 -ms 2 -tickfs 6` 
* 48-hour forecast: \
`verif input1.nc input2.nc -m timeseries -o 48 -f Timeseries_48h.png -fs 12,6 -lw 0.25 -ms 2 -tickfs 6`
* 72-hour forecast: \
`verif input1.nc input2.nc -m timeseries -o 72 -f Timeseries_72h.png -fs 12,6 -lw 0.25 -ms 2 -tickfs 6`
* Bias over time: \
`verif input1.nc input2.nc -m bias -x time -l 601 -f STNS_601Biafor_over_time.png -fs 12,6 -tickfs 8 -lw 1`
