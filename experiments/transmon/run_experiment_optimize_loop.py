from slab.experiments.PulseExperiments_M8195A_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_M8195A_PXI.pulse_experiment_with_switch import Experiment
from slab.experiments.PulseExperiments_M8195A_PXI.sequential_experiment_pxi_with_switch import SequentialExperiment
import json

import os
import numpy as np
from h5py import File
from scipy.optimize import curve_fit
path = os.getcwd()


def take_controls_and_measure(times, controls, taus, acq_num=5000, rb=False, 
                              base_p=0.99913, clifford_index=3, use_vars=False):
    filepath = "S:\\KevinHe\\Optimal Control and Blockade\\Aditya work\\230502_hardware_looping\\"
    path = os.getcwd()
    print("path inside method:", path)
    print("RUNNING HERE")
    ##### Create file that will be accessed by the experiment code #####
    total_time = times[-1]
    measure_pulse_fracs = []
    # taus = [len(times)]  # comment out later
    for tau in taus:
        measure_pulse_fracs.append(min(1.0, times[tau - 1] / total_time))
    print("measuring:", measure_pulse_fracs)
    steps = len(times)
    uneven_tlist = True  # assume for generality that times is not equally spaced, should still work
    # even if times is evenly spaced

    file_number = 0
    pulse_filename = "hardware_looping_tests.h5"
    while os.path.exists(os.path.join(filepath, str(file_number).zfill(5) + "_" + pulse_filename)):
        file_number += 1
    pulse_filename = str(file_number).zfill(5) + "_" + pulse_filename 

    with File(filepath + pulse_filename, 'w') as hf:
        hf.create_dataset('uks', data=np.array([np.array(controls).T]))
        hf.create_dataset('total_time', data=total_time)
        hf.create_dataset('steps', data=steps)
        if uneven_tlist:
            hf.create_dataset('times', data=times)

    ##### Load system parameters #####
    with open('quantum_device_config.json', 'r') as f:
        quantum_device_cfg  = json.load(f)
    with open('experiment_config.json', 'r') as f:
        experiment_cfg = json.load(f)
    with open('hardware_config.json', 'r') as f:
        hardware_cfg = json.load(f)

    ##### Run the experiment which will generate data files #####
    if rb:  # trying an interleaved randomized benchmarking approach
        experiment_name = 'sequential_randomized_benchmarking'
    else:  # just looking at fidelities
        experiment_name = 'optimal_control_test_1step'

    show = 'I'
    TRIGGER_TIME = 1000

    final_output = []
    test_e_evol = [False, True]

    for measure_pulse_frac in measure_pulse_fracs:
        data_filenames = []
        if rb:
            experiment_cfg['randomized_benchmarking']['filename'] = filepath + pulse_filename
            experiment_cfg['randomized_benchmarking']['clifford_index'] = clifford_index
            experiment_cfg['randomized_benchmarking']['interleaved'] = True
            experiment_cfg['randomized_benchmarking']['acquisition_num'] = acq_num
            hardware_cfg['trigger']['period_us'] = TRIGGER_TIME
            sexp = SequentialExperiment(quantum_device_cfg,experiment_cfg,hardware_cfg,experiment_name,path, analyze=False,show=True,
                            data_path="S:/_Data/2021-10-22 Multimode cooldown 16 with JPA as of 2022-05-04/", return_filename=True)
            data_filenames.append(sexp.filename)
        else:
            for setting in test_e_evol:
                if use_vars:
                    experiment_cfg['optimal_control_test_1step']['singleshot'] = True
                experiment_cfg['optimal_control_test_1step']['filename'] = filepath + pulse_filename
                experiment_cfg['optimal_control_test_1step']['pulse_frac'] = measure_pulse_frac
                experiment_cfg['optimal_control_test_1step']['test_e_evol_in_qubit_tom'] = setting
                experiment_cfg['optimal_control_test_1step']['acquisition_num'] = acq_num
                hardware_cfg['trigger']['period_us'] = TRIGGER_TIME
                ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg,plot_visdom=True)
                sequences = ps.get_experiment_sequences(experiment_name)
                print("Sequences generated")
                # print(sequences)
                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name, hvi=False)
                I, Q, data_filename = exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False,
                                                            data_file_path="S:\\_Data\\2021-10-22 Multimode cooldown 16 with JPA as of 2022-05-04\\",
                                                            return_filename=True)
                exp.post_analysis(experiment_name, P=show, show=False)
                data_filenames.append(data_filename)

        ##### Get data from the files #####
        Is = []
        for data_filename in data_filenames:
            with File(data_filename, 'r') as a:
                # hardware_cfg =  (json.loads(a.attrs['hardware_cfg']))
                # experiment_cfg =  (json.loads(a.attrs['experiment_cfg']))
                # quantum_device_cfg = (json.loads(a.attrs['quantum_device_cfg']))
                # expt_cfg = (json.loads(a.attrs['experiment_cfg']))[experiment_names[0]]
                I, Q = np.array(a['I']), np.array(a['Q'])
            data_to_look_at = I
            if rb:
                data_to_look_at = np.mean(data_to_look_at, axis=0)
            Is.append(data_to_look_at)
        Is = np.array(Is)

        ##### Data analysis and manipulation #####
        if rb:
            # could consider removing b as a fit parameter and replace with expected value of 0.5
            def rb_fidelity(x, alpha, a, b):
                return a * alpha ** x + 0.5
            
            expt_cfg = experiment_cfg['randomized_benchmarking']
            n = np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])
            P = Is[0]  # hard-coded, just look at single time (final time)
            popt, pcov = curve_fit(rb_fidelity, n, P, p0=[0.9, -0.5, 0.5], bounds=([0,-1,-1], [1,1,1]))
            if use_vars:
                var_list = pcov[0][0]
            dim = 2
            alpha = popt[0]
            rc_est = (dim - 1) * (1 - min(alpha, base_p) / base_p) / dim  # estimated gate error
            final_output.append(1 - 2*rc_est)  # ideally rc_est will be 0, in worst case is approx 0.5
        else:
            if use_vars:
                Is_new = []  # renormalize singleshot data to 0 and 1
                var_list = []  # list of variances
                for Isingle in Is:
                    Pss_work = Isingle[:, 1:] # first point is always crazy
                    gv = np.mean(Pss_work[-2])
                    ev = np.mean(Pss_work[-1])
                    Pss_use = (np.array(Pss_work[:-2]) - gv) / (ev - gv)  # renormalize
                    Ps = np.mean(Pss_use, axis=1)  # array is meas_num x acq_num
                    var_list.append(np.var(Pss_use, axis=1))
                    Is_new.append(Ps)
                Is = Is_new
            # print(Is)
            # print(data_filenames)
            e_pop_g = Is[-2][2]  # final excited state population at final measurement time, for starting state of g
            x_pauli_g = Is[-2][1]
            y_pauli_g = Is[-2][0]

            e_pop_e = Is[-1][2]  # final excited state population at final measurement time, for starting state of g
            x_pauli_e = Is[-1][1]
            y_pauli_e = Is[-1][0]
            output = [x_pauli_g, y_pauli_g, e_pop_g, x_pauli_e, y_pauli_e, e_pop_e]
            final_output.append(output)
        final_output = np.clip(final_output, 0, 1)  # keep values between 0 and 1
        if use_vars:
            return final_output, var_list

    return np.array(final_output)  # remove [0] later

# if __name__ == "__main__":
#     # Testing that the code runs
#     filename = "S:\\KevinHe\\Optimal Control and Blockade\\Aditya work\\230502_pulse\\converted\\5-2-2023-transmon_no_int_a_iter_3000_time_40.0ns_pinq_true_00001.h5"
#     with File(filename, 'r') as f:
#         # print(f.keys())
#         controls = np.array(f['uks'][()][-1]).T  # controls get transposed later (since that is format output by julia, so add transpose here so the controls are correctly input)
#         # times = f['times'][()]
#         total_time = f['total_time'][()]
#         times = np.linspace(0, total_time, f['steps'][()])
#     # print("Test print statement")
#     print(take_controls_and_measure(times, controls, [len(times)], rb=True, acq_num=500))
   