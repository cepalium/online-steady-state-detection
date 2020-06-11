from ossdem import Signal, TTest

def demo_t_test():
    batch_size = 50
    signal = Signal(bias='linear', bias_params=[1, 200], noise='ar0', noise_params=[0, 0.1], n=500)
    signal.generate()
    signal.plot_signal()
    batches = signal.to_batch(batch_size)
    
    T_crit = 1.2
    t_test = TTest(T_crit=T_crit)
    print("Batch size = {}, T_crit = {}".format(batch_size, T_crit))
    for i, batch in enumerate(batches):
        t_test.insert(batch)
        T_hat = t_test.steady_state_start_point()
        print("Batch {} ({}-{}) - Detected steady state start point = {}".format(i, i*batch_size, (i+1)*batch_size, T_hat))

if __name__ == "__main__":
    demo_t_test()