from RJMC_2CLJ_AUA_Q import parameter_unpacker,\
    rhol_hat_models, Psat_hat_models, SurfTens_hat_models,\
    compound_2CLJ,\
    thermo_data_rhoL, thermo_data_Pv,thermo_data_SurfTens


from RJMC_2CLJ_AUA_Q import sample_from_prior_model_0, sample_from_prior_model_1, sample_from_prior_model_2
prior_samplers = [
    sample_from_prior_model_0,
    sample_from_prior_model_1,
    sample_from_prior_model_2,
]

def property_calculator_rhol(theta, model=0):
    (eps, sig, L, Q) = parameter_unpacker(theta, model=model)
    return rhol_hat_models(compound_2CLJ, thermo_data_rhoL[:, 0], model=model, eps=eps, sig=sig, L=L, Q=Q)  # [kg/m3]


def property_calculator_Psat(theta, model=0):
    (eps, sig, L, Q) = parameter_unpacker(theta, model=model)
    return Psat_hat_models(compound_2CLJ, thermo_data_Pv[:, 0], model=model, eps=eps, sig=sig, L=L, Q=Q)  # [kPa]


def property_calculator_SurfTens(theta, model=0):
    (eps, sig, L, Q) = parameter_unpacker(theta, model=model)
    return SurfTens_hat_models(compound_2CLJ, thermo_data_SurfTens[:, 0], model=model, eps=eps, sig=sig, L=Q, Q=Q)


property_calculators = [
    property_calculator_rhol,
    property_calculator_Psat,
    property_calculator_SurfTens,
]

import numpy as np
nans_encountered = np.zeros((3,3))
from tqdm import tqdm
n_samples = 100000

for model in range(3):
    print('model {}'.format(model))
    for property in range(3):
        print('\tproperty {}'.format(property))
        for _ in tqdm(range(n_samples)):
            theta = prior_samplers[model]()

            nan =  np.isnan(property_calculators[property](theta, model)).any()
            nans_encountered[model, property] += nan
            if nan:
                print('\t\tnan encountered! ', theta)

print(100 * nans_encountered / n_samples)
