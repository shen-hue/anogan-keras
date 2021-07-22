import numpy as np


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()
    freq = np.zeros((no,dim))
    phase = np.zeros((no,dim))

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.8)
            phase = np.random.uniform(0, 0.8)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return np.asarray(data)


def anomaly_sine_data_generation(no, seq_len, dim):
    """Sine data generation.

  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

  Returns:
    - data: generated data
  """
    # Initialize the output
    data = list()
    freq = np.zeros((no,dim))
    phase = np.zeros((no,dim))

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.8)
            phase = np.random.uniform(0, 0.8)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)
        # add anomaly points
        for ind in range(3):         # add anomaly in 8 features
            # change_index = np.random.randint(seq_len - 1, size=3)      # choose the anomaly point randomly
            change_index = [2, 12, 22]           # choose the anomaly point fixed
            for m in range(3):
                temp[ind][change_index[m]:change_index[m]+2] = np.full(2,2)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)
    print('')

    return np.asarray(data)
