"""Contains methods for performing n-gate tomography"""
import numpy as np


def angle_sets_to_evaluate(num_params):
    """Return the angles at which the expectation values
    are to be measured for num_param gate tomography

    Args:
        num_params (int): Number of rotation gates the
            tomography is with respect to

    Returns:
        np.ndarray([3**num_params,num_params]): Ordered
            array of parameters values
    """
    angles = np.zeros([3**num_params, num_params])
    for i in range(3**num_params):
        base_3_str = np.base_repr(i, 3).zfill(num_params)
        for j, ind in zip(range(num_params), base_3_str):
            if ind == "0":
                angles[i, j] = -np.pi / 2
            elif ind == "1":
                angles[i, j] = 0
            elif ind == "2":
                angles[i, j] = np.pi / 2
    return angles


def measurements_to_zero_delta_pi_bases(measurements):
    """Tomography requires expactation values when each
        angle is either 0, np.pi/2,-np.pi/2, and np.pi.
        This method calculates the value for np.pi from
        the other 3 values and arranges the data in
        appropriate form

    Args:
        measurements (np.ndarray): The expectation values
        with angles obtained from angle_sets_to_evaluate

    Returns:
        np.ndarray: New expectation value measurements
    """
    num_params = int(np.log(len(measurements)) / np.log(3))
    new_measurements = np.array(measurements)
    for j in range(num_params):
        for i in range(3 ** (num_params - 1)):
            if num_params == 1:
                base_3_str = ""
            else:
                base_3_str = np.base_repr(i, 3).zfill(num_params - 1)
            l_str = base_3_str[: num_params - (j + 1)]
            r_str = base_3_str[num_params - (j + 1) :]
            ind_0 = int(l_str + "0" + r_str, 3)
            ind_1 = int(l_str + "1" + r_str, 3)
            ind_2 = int(l_str + "2" + r_str, 3)

            val_minus_pi_by_2 = new_measurements[ind_0]
            val_0 = new_measurements[ind_1]
            val_pi_by_2 = new_measurements[ind_2]

            new_measurements[ind_0] = val_0
            new_measurements[ind_1] = val_pi_by_2 - val_minus_pi_by_2
            new_measurements[ind_2] = (val_pi_by_2 + val_minus_pi_by_2) - val_0

    return new_measurements


def reconstructed_cost(angles, measurements):
    """Calculate the cost from the tomography-reconstructed cost function

    Args:
        angles (np.ndarray): Angles to evaluate expectation value at
        measurements (np.ndarray): Expectation values of tomography measurements

    Returns:
        float: Expectation value
    """
    total = 0
    num_params = len(angles)
    for i in range(3**num_params):
        product = 1
        product *= measurements[i]
        base_3_str = np.base_repr(i, 3).zfill(num_params)
        for j in range(num_params):
            angle = angles[j] / 2
            if base_3_str[j] == "0":
                product *= np.cos(angle) * np.cos(angle)
            elif base_3_str[j] == "1":
                product *= np.cos(angle) * np.sin(angle)
            elif base_3_str[j] == "2":
                product *= np.sin(angle) * np.sin(angle)
        total += product
    return total
