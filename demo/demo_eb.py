from ossdem import Signal, ExactBayes

def demo_eb():
    batch_size = 50
    signal = Signal(bias='linear', bias_params=[1, 200], noise='ar0', noise_params=[0, 0.1], n=500)
    signal.generate()
    signal.plot_signal()
    batches = signal.to_batch(batch_size)

    m = 20
    s_0 = 0.002
    eb = ExactBayes(m=m, s_0=s_0)
    print("Batch size = {}, m = {}, s_0 ={}".format(batch_size, m, s_0))
    for i, batch in enumerate(batches):
        eb.insert(batch)
        T_hat = eb.steady_state_start_point()
        print("Batch {} ({}-{}) - Detected steady state start point = {}".format(i, i*batch_size, (i+1)*batch_size, T_hat))

if __name__ == "__main__":
    demo_eb()