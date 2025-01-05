import matplotlib.pyplot as plt


def make_ioplots(
        data,
        input_names,
        measurement_names,
        available_inputs,
        available_measurements,
        time_name='time',
        time_range=None,
        var_rename_map=None,
    ):

    if time_range is None:
        time_range = slice(None, None)
    if var_rename_map is None:
        var_rename_map = {name: name for name in input_names + measurement_names}
    
    ny = len(measurement_names)
    nu = len(input_names)

    fig, axes = plt.subplots(ny + nu, 1, sharex=True, figsize=(7, 0.5 + 1.5 * (ny + nu)))

    for i, (ax, name) in enumerate(zip(axes[:ny], measurement_names)):
        x = data.set_index(time_name).loc[time_range, name]
        unit = available_measurements[name]['Unit']
        if unit == 'K':
            x = x - 273.15
            unit = 'deg C'
        x.plot(ax=ax, label=var_rename_map[name])
        label = "$y_{%d}$" % (i+1) + f" ({unit})"
        ax.set_ylabel(label)
        ax.set_title(var_rename_map[name])
        ax.grid()

    for i, (ax, name) in enumerate(zip(axes[ny:], input_names)):
        x = data.set_index(time_name).loc[time_range, name]
        unit = available_inputs[name]['Unit']
        x.plot(ax=ax, drawstyle="steps-post", label=var_rename_map[name])
        label = "$u_{%d}$" % (i+1) + f" ({unit})"
        ax.set_ylabel(label)
        ax.set_title(var_rename_map[name])
        ax.grid()

    axes[-1].set_xlabel('Time (days)')

    return fig, axes


def make_ioplots_combined(
        data,
        input_names,
        measurement_names,
        available_inputs,
        available_measurements,
        time_name='time',
        time_range=None,
        var_rename_map=None,
    ):

    if time_range is None:
        time_range = slice(None, None)
    if var_rename_map is None:
        var_rename_map = {name: name for name in input_names + measurement_names}

    ny = len(measurement_names)
    nu = len(input_names)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 4.5))

    ax = axes[0]
    x = (
        data.set_index(time_name)
        .loc[time_range, measurement_names]
        .rename(columns=var_rename_map)
    )
    x.plot(ax=ax)
    if ny == 1:
        label = "$y_{1}$"
    else:
        label = "$y_{1-%d}$" % ny
    ax.set_ylabel(label)
    ax.set_title("Outputs")
    ax.grid()

    ax = axes[1]
    x = (
        data.set_index(time_name)
        .loc[time_range, input_names]
        .rename(columns=var_rename_map)
    )
    x.plot(ax=ax, drawstyle="steps-post")
    if nu == 1:
        label = "$u_{1}$"
    else:
        label = "$u_{1-%d}$" % nu
    ax.set_ylabel(label)
    ax.set_title("Inputs")
    ax.grid()

    axes[-1].set_xlabel('Time (days)')

    return fig, axes