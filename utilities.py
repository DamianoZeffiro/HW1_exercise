from model import *
# The function gradient_descent takes an initial vector y_init and gradient descent optimization parameters
# and runs the gradient descent algorithm to find the optimal y.
def gradient_descent(y_init, H, b, true_labels_unlabelled, min_loss = -np.inf, num_iterations=1000, grad_tol=1e-1):
    y = y_init
    losses = []  # List to record the loss at each iteration
    accuracies = []  # List to record the accuracy at each iteration
    cpu_times = []  # List to record the computation time at each iteration
    start_time = time.process_time()  # Start the timer

    # The main loop for the gradient descent algorithm
    for i in range(num_iterations):
        #TODO: write gradient descent algorithm with linesearch and update losses, accuracies and cpu_times

        #TODO: insert the missing variables in the stopping criteraia
        if min_loss != -np.inf:
            if ... < min_loss:  # If the loss is less than the minimum allowed loss, break the loop
                break
        else:
            if np.linalg.norm(...) < grad_tol:  # If the norm of the gradient is less than the tolerance, break the loop
                break

    # Return the final y, along with the recorded losses, accuracies, and computation times
    return y, losses, accuracies, cpu_times

# This function plots the optimization metrics for the gradient descent algorithm.
def plot_optimization_metrics(obj_star, y_star, y_gd, losses_gd, accuracies_gd, cpu_times_gd, true_labels_unlabelled):
    plt.figure(figsize=(10, 10))

    # Plot 1: Iteration vs Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(accuracies_gd, label='Gradient Descent')
    plt.axhline(y=np.mean(true_labels_unlabelled == np.sign(y_star)), color='r', linestyle='--',
                label='Numpy Solve')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Iteration vs Accuracy')
    plt.legend()

    # Plot 2: Iteration vs Loss
    plt.subplot(2, 2, 2)
    plt.plot(losses_gd, label='Gradient Descent')
    plt.axhline(y=obj_star, color='r', linestyle='--', label='Correct solution')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Iteration vs Loss')
    plt.legend()

    # Plot 3: CPU Time vs Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(cpu_times_gd, accuracies_gd, label='Gradient Descent')
    plt.axhline(y=np.mean(true_labels_unlabelled == np.sign(y_star)), color='r', linestyle='--', label = 'Correct solution')
    plt.xlabel('CPU Time')
    plt.ylabel('Accuracy')
    plt.title('CPU Time vs Accuracy')
    plt.legend()

    # Plot 4: CPU Time vs Loss
    plt.subplot(2, 2, 4)
    plt.plot(cpu_times_gd, losses_gd, label='Gradient Descent')
    plt.axhline(y=obj_star, color='r', linestyle='--', label='Correct solution')
    plt.xlabel('CPU Time')
    plt.ylabel('Loss')
    plt.title('CPU Time vs Loss')
    plt.legend()

    plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
    plt.show()  # Display the figure