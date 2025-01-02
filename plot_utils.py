import matplotlib.pyplot as plt


def make_ioplots(
        data, 
        input_names, 
        measurement_names, 
        available_inputs, 
        available_measurements, 
        time_range=None
    ):

    if time_range is None:
        time_range = slice(None, None)
    
    ny = len(measurement_names)
    nu = len(input_names)

    fig, axes = plt.subplots(ny + nu, 1, sharex=True, figsize=(7, 0.5 + 1.5 * (ny + nu)))

    for i, (ax, name) in enumerate(zip(axes[:ny], measurement_names)):
        x = data.set_index('time_days').loc[time_range, name]
        unit = available_measurements[name]['Unit']
        if unit == 'K':
            x = x - 273.15
            unit = 'deg C'
        x.plot(ax=ax)
        label = "$y_{%d}$" % (i+1) + f" ({unit})"
        ax.set_ylabel(label)
        ax.set_title(available_measurements[name]['Description'][:48])
        ax.grid()

    for i, (ax, name) in enumerate(zip(axes[ny:], input_names)):
        x = data.set_index('time_days').loc[time_range, name]
        unit = available_inputs[name]['Unit']
        x.plot(ax=ax, drawstyle="steps-post")
        label = "$u_{%d}$" % (i+1) + f" ({unit})"
        ax.set_ylabel(label)
        ax.set_title(available_inputs[name]['Description'][:48])
        ax.grid()

    axes[-1].set_xlabel('Time (days)')

    return fig, axes


def make_ioplots_combined(
        data, 
        input_names, 
        measurement_names, 
        available_inputs, 
        available_measurements, 
        time_range=None
    ):

    if time_range is None:
        time_range = slice(None, None)

    ny = len(measurement_names)
    nu = len(input_names)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 4.5))

    ax = axes[0]
    x = data.set_index('time_days').loc[time_range, measurement_names]
    x.plot(ax=ax)
    if ny == 1:
        label = "$y_{1}$"
    else:
        label = "$y_{1-%d}$" % ny
    ax.set_title("Outputs")
    ax.grid()

    ax = axes[1]
    x = data.set_index('time_days').loc[time_range, input_names]
    x.plot(ax=ax, drawstyle="steps-post")
    if nu == 1:
        label = "$y_{1}$"
    else:
        label = "$y_{1-%d}$" % nu
    ax.set_ylabel(label)
    ax.set_title("Inputs")
    ax.grid()

    axes[-1].set_xlabel('Time (days)')

    return fig, axes