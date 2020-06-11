from ossdem import Signal, SlopeDetectionMethod

def demo_SDM():
    batch_size = 50
    signal = Signal(bias='linear', bias_params=[1, 200], noise='ar0', noise_params=[0, 0.1], n=500)         # slope_crit 0.001
    signal.generate()
    signal.plot_signal()
    batches = signal.to_batch(batch_size)

    slope_crit = 0.002
    slope_detector = SlopeDetectionMethod(slope_crit=slope_crit)
    print("Batch size = {}, slope_crit = {}".format(batch_size, slope_crit))
    # 
    for i, batch in enumerate(batches):
        slope_detector.insert(batch)
        T_hat = slope_detector.steady_state_start_point()
        print("Batch {} ({}-{}) - Detected steady state start point = {}".format(i, i*batch_size, (i+1)*batch_size, T_hat))

if __name__ == "__main__":
    demo_SDM()