
#!/bin/bash

python rjmc_simulation_refit_prior.py -c 'C2H6' -s 1000000 -p 'rhol+Psat' -l 'BAR_test' &
python rjmc_simulation_refit_prior.py -c 'C2H6' -s 1000000 -p 'All' -l 'BAR_test' &
python rjmc_simulation_refit_prior.py -c 'O2' -s 1000000 -p 'rhol+Psat' -l 'BAR_test' &
python rjmc_simulation_refit_prior.py -c 'O2' -s 1000000 -p 'All' -l 'BAR_test' &
python rjmc_simulation_refit_prior.py -c 'N2' -s 1000000 -p 'rhol+Psat' -l 'BAR_test' &
python rjmc_simulation_refit_prior.py -c 'N2' -s 1000000 -p 'All' -l 'BAR_test'




