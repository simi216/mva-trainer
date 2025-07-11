for hidden_size in [16, 32, 64]:
    for attention_blocks in [1, 2, 3]:
        for ff_layers in [1, 2]:
            for attention_heads in [4, 8, 12]:
                for dropout in [0.05, 0.1]:
                    for regularization_lambda in [0.001, 0.01]:
                        for alpha in [0.5, 1]:
                            cmd = f"{hidden_size} {attention_blocks} {ff_layers} {attention_heads} {dropout} {regularization_lambda} {alpha}"
                            print(cmd)