for i in range(2,5):
    for j in range(3, 10):
        for k in range(0, 6, 2):
            for dropout in [0, 0.05, 0.1, 0.2]:
                cmd = f"{i} {j} {k} {dropout}"
                print(cmd)