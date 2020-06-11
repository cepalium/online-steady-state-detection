from ossdem import Signal
import matplotlib.pyplot as plt

def demo_signal():
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(13, 9))

    # header titles per row & per column
    cols = ['No noise', 'AR0', 'AR1']
    rows = ['Linear', 'Quadratic', 'Exponential', 'Oscillating']

    for a, col in zip(ax[0], cols):
        a.set_title(col)

    for a, row in zip(ax[:,0], rows):
        a.set_ylabel(row)

    # plot subplots
    signal = Signal(bias='linear', bias_params=[1, 200], noise='ar0', noise_params=[0, 0], n=500)
    y = signal.generate()
    ax[0, 0].plot(y, 'b', linewidth=1)

    signal = Signal(bias='linear', bias_params=[1, 200], noise='ar0', noise_params=[0, 0.1], n=500)
    y = signal.generate()
    ax[0, 1].plot(y, 'b', linewidth=1)

    signal = Signal(bias='linear', bias_params=[1, 200], noise='ar1', noise_params=[0, 0.1, 0.4], n=500)
    y = signal.generate()
    ax[0, 2].plot(y, 'b', linewidth=1)

    signal = Signal(bias='quadratic', bias_params=[1, 200], noise='ar0', noise_params=[0, 0], n=500)
    y = signal.generate()
    ax[1, 0].plot(y, 'b', linewidth=1)

    signal = Signal(bias='quadratic', bias_params=[1, 200], noise='ar0', noise_params=[0, 0.1], n=500)
    y = signal.generate()
    ax[1, 1].plot(y, 'b', linewidth=1)

    signal = Signal(bias='quadratic', bias_params=[1, 200], noise='ar1', noise_params=[0, 0.1, 0.4], n=500)
    y = signal.generate()
    ax[1, 2].plot(y, 'b', linewidth=1)

    signal = Signal(bias='exponential', bias_params=[1, 200], noise='ar0', noise_params=[0, 0], n=500)
    y = signal.generate()
    ax[2, 0].plot(y, 'b', linewidth=1)

    signal = Signal(bias='exponential', bias_params=[1, 200], noise='ar0', noise_params=[0, 0.1], n=500)
    y = signal.generate()
    ax[2, 1].plot(y, 'b', linewidth=1)

    signal = Signal(bias='exponential', bias_params=[1, 200], noise='ar1', noise_params=[0, 0.1, 0.4], n=500)
    y = signal.generate()
    ax[2, 2].plot(y, 'b', linewidth=1)

    signal = Signal(bias='oscillating', bias_params=[1, 200, 30], noise='ar0', noise_params=[0, 0], n=500)
    y = signal.generate()
    ax[3, 0].plot(y, 'b', linewidth=1)

    signal = Signal(bias='oscillating', bias_params=[1, 200, 30], noise='ar0', noise_params=[0, 0.1], n=500)
    y = signal.generate()
    ax[3, 1].plot(y, 'b', linewidth=1)

    signal = Signal(bias='oscillating', bias_params=[1, 200, 30], noise='ar1', noise_params=[0, 0.1, 0.4], n=500)
    y = signal.generate()
    ax[3, 2].plot(y, 'b', linewidth=1)

    # show plots
    ax[0, 0].set_ylim(-0.2, 1.4)
    ax[1, 0].set_ylim(-0.2, 1.4)
    plt.show()

if __name__ == "__main__":
    demo_signal()