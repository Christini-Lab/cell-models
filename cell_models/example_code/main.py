"""Main driver for program. Before running, make sure to have a directory
called `figures` for matplotlib pictures to be stored in."""
import matplotlib.pyplot as plt
from os import listdir, mkdir
import pickle

from cell_models import protocols
from cell_models.kernik import KernikModel
from cell_models.paci_2018 import PaciModel
from cell_models.ohara_rudy import OharaRudyModel


# Spontaneous
def spontaneous_example():
    """
    Plots spontaneous Kernik/Paci and stimulated O'Hara Rudy
    """
    KERNIK_PROTOCOL = protocols.SpontaneousProtocol(2000)
    kernik_baseline = KernikModel()
    tr_b = kernik_baseline.generate_response(KERNIK_PROTOCOL,
            is_no_ion_selective=False)
    plt.plot(tr_b.t, tr_b.y)
    plt.show()

    #PACI_PROTOCOL = protocols.SpontaneousProtocol(2000)
    #paci_baseline = PaciModel()
    #tr_bp = paci_baseline.generate_response(PACI_PROTOCOL,
    #        is_no_ion_selective=False)
    #plt.plot(tr_bp.t, tr_bp.y)
    #plt.show()

    ##HAVE NOT SET UP ARTEFACT OR PARAMETER SETTING
    #OHARA_RUDY = protocols.PacedProtocol(model_name="OR")
    #or_baseline = OharaRudyModel()
    #tr = or_baseline.generate_response(OHARA_RUDY, is_no_ion_selective=False)
    #plt.plot(tr.t, tr.y)
    #plt.show()

# Paced 
def stimulated_example():
    KERNIK_PROTOCOL = protocols.PacedProtocol(model_name="Kernik")
    kernik_baseline = KernikModel()
    tr_b = kernik_baseline.generate_response(KERNIK_PROTOCOL,
            is_no_ion_selective=False)
    plt.plot(tr_b.t, tr_b.y)
    plt.show()

    #PACI_PROTOCOL = protocols.PacedProtocol(model_name="Paci")
    #paci_baseline = PaciModel()
    #tr_bp = paci_baseline.generate_response(PACI_PROTOCOL,
    #        is_no_ion_selective=False)
    #plt.plot(tr_bp.t, tr_bp.y)
    #plt.show()

    ##HAVE NOT SET UP ARTEFACT OR PARAMETER SETTING
    #OHARA_RUDY = protocols.PacedProtocol(model_name="OR")
    #or_baseline = OharaRudyModel()
    #tr = or_baseline.generate_response(OHARA_RUDY, is_no_ion_selective=False)
    #plt.plot(tr.t, tr.y)
    #plt.show()

# Set parameter values (Kernik) and visualize currents
def update_params():
    KERNIK_PROTOCOL = protocols.PacedProtocol(model_name="Kernik", stim_end=10000, stim_mag=2)
    mod_adjusted = KernikModel(updated_parameters={
                'G_K1': 0, # Set Kernik model IK1 to Zero
                'G_K1_Ishi': .8,
                'G_Kr': .2,
                'G_Ks': .5,
                'G_to': .25,
                'P_CaL': 2.25,
                'G_CaT': 1,
                'G_Na': 2,
                'G_F': .4,
                'K_NaCa': 1,
                'P_NaK': 1,
                'VmaxUp': 1,
                'V_leak': 1,
                'ks': 1,
                'G_b_Na': 1,
                'G_b_Ca': 1,
                'G_PCa': 1,
                'G_seal_leak': 1,
                'V_off': 1
            })
    tr = mod_adjusted.generate_response(KERNIK_PROTOCOL,
            is_no_ion_selective=False)
    tr.plot_with_individual_currents()
    plt.show()

    tr.plot_with_individual_currents(['I_Na', 'I_Kr', 'I_Ks', 'I_CaL'])
    plt.show()

# Dynamically clamp IK1
def dynamic_ik1_ishi():
    KERNIK_PROTOCOL = protocols.PacedProtocol(model_name="Kernik", stim_end=10000, stim_mag=2)

    #Baseline Kernik
    baseline_kernik = KernikModel()
    tr_baseline = baseline_kernik.generate_response(KERNIK_PROTOCOL,
            is_no_ion_selective=False)

    #Baseline Kernik with Ishihara DC
    mod_with_ishi_dc = KernikModel(no_ion_selective_dict={'I_K1_Ishi': .5})
    tr_ishi_dc = mod_with_ishi_dc.generate_response(KERNIK_PROTOCOL,
        is_no_ion_selective=True)

    #Kernik with Ishihara IK1 and Ishihara DC
    mod_ishi_and_dc = KernikModel(updated_parameters={'G_K1': 0, 'G_K1_Ishi': .5},
            no_ion_selective_dict={'I_K1_Ishi': .5})
    tr_ishi_and_dc = mod_ishi_and_dc.generate_response(KERNIK_PROTOCOL,
        is_no_ion_selective=True)

    # Compare the three
    plt.plot(tr_baseline.t, tr_baseline.y, label="Baseline") 
    plt.plot(tr_ishi_dc.t, tr_ishi_dc.y, label="Ishihara DC") 
    plt.plot(tr_ishi_and_dc.t, tr_ishi_and_dc.y,
            label="Ishihara IK1 + Ishihara DC") 
    plt.legend()
    plt.show()

# Incrementally increase IK1 ishihara current
def increment_ishi():
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    KERNIK_PROTOCOL = protocols.SpontaneousProtocol(6000)
    baseline_kernik = KernikModel()
    tr = baseline_kernik.generate_response(KERNIK_PROTOCOL,
                is_no_ion_selective=False)
    
    axs[0].plot(tr.t, tr.y, color='k')
    axs[1].plot(tr.t, tr.current_response_info.get_current_summed(), color='k')
    axs[2].plot(tr.t, tr.current_response_info.get_current('I_K1_Ishi'),
            color='k', label=f'Baseline')

    for i in range(0, 10):
        gk1 = i/10
        mod = KernikModel(updated_parameters={'G_K1': 0,
            'G_K1_Ishi': gk1})
        tr = mod.generate_response(KERNIK_PROTOCOL,
                is_no_ion_selective=False)

        axs[0].plot(tr.t, tr.y)
        axs[1].plot(tr.t, tr.current_response_info.get_current_summed())
        axs[2].plot(tr.t, tr.current_response_info.get_current('I_K1_Ishi'),
                label=f'IK1_ishi={gk1}')

    axs[0].set_ylabel("Vm", fontsize=14)
    axs[1].set_ylabel("Im", fontsize=14)
    axs[2].set_ylabel("IK1 Ishi", fontsize=14)
    axs[2].set_xlabel("Time (ms)", fontsize=14)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

    plt.legend()
    plt.show()

# Set the intracellular Ki and Nai concentrations
def set_intracellular_conc():
    KERNIK_PROTOCOL = protocols.PacedProtocol(model_name="Kernik", stim_end=10000, stim_mag=2)
    KERNIK_PROTOCOL = protocols.SpontaneousProtocol(6000)
    kernik_baseline = KernikModel(updated_parameters={'G_Kr': 3})
    tr_b = kernik_baseline.generate_response(KERNIK_PROTOCOL,
            is_no_ion_selective=False)

    kernik_constant_conc = KernikModel(updated_parameters={'G_Kr': 3, 'G_K1': 0},
            nai_millimolar=10, ki_millimolar=130)
    tr_conc = kernik_constant_conc.generate_response(KERNIK_PROTOCOL,
            is_no_ion_selective=False)

    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    names = ['Baseline', 'Constant Concentrations']
    mods = [kernik_baseline, kernik_constant_conc]

    for i, tr in enumerate([tr_b, tr_conc]):
        axs[0].plot(tr.t, tr.y)
        axs[1].plot(mods[i].t, mods[i].y[:, 3])
        axs[2].plot(mods[i].t, mods[i].y[:, 4], label=names[i])

    axs[0].set_ylabel("Vm", fontsize=14)
    axs[1].set_ylabel("Nai", fontsize=14)
    axs[2].set_ylabel("Ki", fontsize=14)
    axs[2].set_xlabel("Time (ms)", fontsize=14)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

    plt.legend()
    plt.show()

#TODO: Update artefacts for VC protocol
def update_artefacts():
    mod_with_artefact = KernikModel(is_exp_artefact=True,
                                           exp_artefact_params={
                                               'g_leak': 1, #1/GOhm
                                               'e_leak': 0, #reversal leak
                                               'v_off': -2.8, #ljp
                                               'c_p': 4, #NOT USED
                                               'r_pipette': 2E-3, #pip res GOhm
                                               'c_m': 60, #pF
                                               'r_access': 25E-3, # ser res GOhm
                                               'alpha': .85 # series comp
                                               })

# Run parameter tuning experiment to fit specified conductances
def run_parameter_tuning():
    from cell_models.ga.parameter_tuning import ParameterTuningGeneticAlgorithm
    from cell_models.ga import ga_configs, target_objective, mp_parameter_tuning

    KERNIK_PARAMETERS = [
        ga_configs.Parameter(name='G_Na', default_value=1),
        ga_configs.Parameter(name='G_F', default_value=1),
        ga_configs.Parameter(name='G_Ks', default_value=1),
        ga_configs.Parameter(name='G_Kr', default_value=1),
        ga_configs.Parameter(name='G_K1', default_value=1),
        ga_configs.Parameter(name='G_b_Na', default_value=1),
        ga_configs.Parameter(name='P_CaL', default_value=1),
        ga_configs.Parameter(name='G_PCa', default_value=1),
        ga_configs.Parameter(name='G_b_Ca', default_value=1),
        ga_configs.Parameter(name='K_NaCa', default_value=1)
    ]

    VC_PROTO = protocols.VoltageClampProtocol()

    KERNIK_MOD = KernikModel(is_exp_artefact=True)

    baseline_target = target_objective.create_target_from_protocol(
            KERNIK_MOD, VC_PROTO)

    vc_config = ga_configs.ParameterTuningConfig(
        population_size=140,
        max_generations=80,
        targets={'baseline_target': baseline_target},
        tunable_parameters=KERNIK_PARAMETERS,
        params_lower_bound=0.1,
        params_upper_bound=10,
        mate_probability=0.9,
        mutate_probability=0.9,
        gene_swap_probability=0.2,
        gene_mutation_probability=0.2,
        tournament_size=2,
        with_exp_artefact=False,
        model_name='Kernik', 
        cell_model=KernikModel)

    mp_parameter_tuning.start_ga(vc_config)


# Run parameter tuning experiment to fit specified conductances
def run_vc_proto_ga():
    from cell_models.ga import ga_configs, mp_voltage_clamp_optimization
    from cell_models.ga import voltage_clamp_optimization_experiments

    WITH_ARTEFACT = True

    VCO_CONFIG = ga_configs.VoltageOptimizationConfig(
        window=2,
        step_size=2,
        steps_in_protocol=5,
        step_duration_bounds=(5, 1000),
        step_voltage_bounds=(-120, 60),
        target_current='I_Na',
        population_size=200,
        max_generations=50,
        mate_probability=0.9,
        mutate_probability=0.9,
        gene_swap_probability=0.2,
        gene_mutation_probability=0.1,
        tournament_size=2,
        step_types=['step', 'ramp'],
        with_artefact=WITH_ARTEFACT,
        model_name='Kernik')

    #The currents parameter here is a vestige of an old implementation. TODO: remove
    COMBINED_VC_CONFIG = ga_configs.CombinedVCConfig(
        currents=['I_Na', 'I_K1', 'I_To', 'I_CaL', 'I_Kr', 'I_Ks'],
        step_range=range(2, 3, 1),
        adequate_fitness_threshold=0.95,
        ga_config=VCO_CONFIG)

    LIST_OF_CURRENTS = ['I_Na', 'I_Kr', 'I_Ks', 'I_F']

    with_artefact=True
    vco_dir_name = f'trial_steps_ramps_{VCO_CONFIG.model_name}_{VCO_CONFIG.population_size}_{VCO_CONFIG.max_generations}_{VCO_CONFIG.steps_in_protocol}_{VCO_CONFIG.step_voltage_bounds[0]}_{VCO_CONFIG.step_voltage_bounds[1]}'

    if not vco_dir_name in listdir('results'):
        mkdir(f'results/{vco_dir_name}')

    for c in LIST_OF_CURRENTS:
        f = f"./results/{vco_dir_name}/ga_results_{c}_artefact_{WITH_ARTEFACT}"
        print(f"Finding best protocol for {c}. Writing protocol to: {f}")
        VCO_CONFIG.target_current = c
        result = mp_voltage_clamp_optimization.start_ga(COMBINED_VC_CONFIG)

        pickle.dump(result, open(f, 'wb'))



def main():
    #spontaneous_example()
    #stimulated_example()
    #update_params()
    #dynamic_ik1_ishi()
    #increment_ishi()
    #set_intracellular_conc()
    #run_parameter_tuning()
    run_vc_proto_ga()


if __name__ == '__main__':
    main()
