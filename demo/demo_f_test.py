from ossdem import Signal, FTest

def demo_f_test():
    batch_size = 100
    signal = Signal(bias='linear', bias_params=[1, 200], noise='ar0', noise_params=[0, 0.1], n=500)
    signal.generate()
    signal.plot_signal()
    batches = signal.to_batch(batch_size)

    F_crit = 1.2
    f_test = FTest(F_crit=F_crit)
    print("Batch size = {}, F_crit = {}".format(batch_size, F_crit))
    for i, batch in enumerate(batches):
        f_test.insert(batch)
        T_hat = f_test.steady_state_start_point()
        print("Batch {} ({}-{}) - Detected steady state start point = {}".format(i, i*batch_size, (i+1)*batch_size, T_hat))

if __name__ == "__main__":
    demo_f_test()