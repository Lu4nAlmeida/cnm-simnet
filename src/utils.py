from simulate_ldpm import ProjectileMotionSimulation
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_global_scalers(num_dim, shape, time_steps, sample_size=2000):
    input_data = []
    output_data = []

    for _ in range(sample_size):
        # Same logic as in generate_data()
        ldpm = ProjectileMotionSimulation()
        ldpm.random_initial_parameters(num_dim, shape)
        ldpm.compute_output(time_steps)

        inputs = np.concatenate([
            ldpm.vel,
            ldpm.acc,
            ldpm.mass[:, :-1],
            ldpm.drg[:, :-1],
        ], axis=1)
        outputs = np.concatenate([
            ldpm.pos,
            ldpm.vel,
            ldpm.frc
        ], axis=1)

        input_data.append(inputs)
        output_data.append(outputs)

    X = np.array(input_data).reshape(-1, num_dim * 2 + 2)
    y = np.array(output_data).reshape(-1, num_dim * 3)

    X_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y)
    return X_scaler, y_scaler
