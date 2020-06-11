from ossdem import Signal, RTest

def demo_r_test():
    batch_size = 50
    signal = Signal(bias='quadratic', bias_params=[1, 200], noise='ar0', noise_params=[0, 0.1], n=500)
    signal.generate()
    signal.plot_signal()
    batches = signal.to_batch(batch_size)
    
    R_crit = 1.1
    l1 = 0.03
    l2 = 0.05
    l3 = 0.05
    r_test = RTest(lambda1=l1, lambda2=l2, lambda3=l3, R_crit=R_crit)
    print("Batch size = {}, R_crit = {}, lambda_1={}, lambda_2={}, lambda_3={}".format(batch_size, R_crit, l1, l2, l3))
    for i, batch in enumerate(batches):
        r_test.insert(batch)
        T_hat = r_test.steady_state_start_point()
        print("Batch {} ({}-{}) - Detected steady state start point = {}".format(i, i*batch_size, (i+1)*batch_size, T_hat))

if __name__ == "__main__":
    demo_r_test()