#this project defines 3 functions to asses the performance of gradient descent and np.linalg.solve for the
#semi-supervised learning problem. given in HW1. See descriptions below
# in this exercise version, there are some incomplete codes in utilities and model

# First, we import the necessary classes or functions from the model module.
from model import *
from utilities import *

# This function is for comparing the time taken by np.linalg.solve and gradient descent for different data sizes
def compare_solvers(vec_num_samples=np.arange(100, 1000, 100), vec_num_labelled=np.arange(10, 100, 10), gamma=3):
    solve_times = []  # List to record the time taken by np.linalg.solve
    gd_times = []  # List to record the time taken by gradient descent

    # Loop over different data sizes
    for i in range(len(vec_num_samples)):
        num_samples = vec_num_samples[i]
        num_labelled = vec_num_labelled[i]
        print(f"Starting iteration {i + 1} with {num_samples} samples and {num_labelled} labelled data points.")

        # Initialize the semi-supervised model
        model = SemisupervisedModel(num_samples=num_samples, num_labelled=num_labelled, gamma=gamma)
        H, g = model.H, model.g

        # Solve the problem using np.linalg.solve and record the time taken
        start_time = time.time()
        y_pred = -np.linalg.solve(H, g)
        solve_time = time.time() - start_time
        solve_times.append(solve_time)
        print(f"np.linalg.solve completed in {solve_time} seconds.")

        # Calculate the minimum objective value (i.e., loss at optimal solution)
        obj_star = 0.5 * np.dot(y_pred.T, np.dot(H, y_pred)) + np.dot(g.T, y_pred)

        # Run gradient descent and record the time taken
        y_init = np.zeros_like(y_pred)
        start_time = time.time()
        y_gd, losses_gd, accuracies_gd, cpu_times_gd = gradient_descent(y_init, H, g, model.true_labels_unlabelled,
                                                                        min_loss=0.99 * obj_star, num_iterations=10000)
        gd_time = time.time() - start_time
        gd_times.append(gd_time)
        print(f"Gradient descent completed in {gd_time} seconds.\n")

    # Plot the time taken by np.linalg.solve and gradient descent as a function of the number of samples
    plt.figure(figsize=(10, 6))
    plt.plot(vec_num_samples, solve_times, label='np.linalg.solve')
    plt.plot(vec_num_samples, gd_times, label='Gradient Descent')
    plt.xlabel('Number of Samples')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Comparison of Time Taken by np.linalg.solve and Gradient Descent')
    plt.show()


# This function is for visualizing the convergence of the gradient descent algorithm
def gradient_descent_convergence(num_samples, num_labelled, gamma):
    # Initialize the semi-supervised model
    model = SemisupervisedModel(num_samples=num_samples, num_labelled=num_labelled, gamma=gamma)
    H, g = model.H, model.g

    # Solve the problem using np.linalg.solve
    y_pred = -np.linalg.solve(H, g)
    # Calculate the optimal objective value
    obj_star = 0.5 * np.dot(y_pred.T, np.dot(H, y_pred)) + np.dot(g.T, y_pred)

    # Initialize the y for gradient descent
    y_init = np.zeros_like(y_pred)
    # Run gradient descent
    y_gd, losses_gd, accuracies_gd, cpu_times_gd = gradient_descent(y_init, H, g, model.true_labels_unlabelled,
                                                                    num_iterations=10000)
    # Plot the metrics of the gradient descent optimization
    plot_optimization_metrics(obj_star, y_pred, y_gd, losses_gd, accuracies_gd, cpu_times_gd,
                              model.true_labels_unlabelled)
    # Plot the clusters formed by the gradient descent solution
    model.plot_clusters(np.sign(y_gd))


# This function is for plotting the solution of the semi-supervised learning problem
def plot_solution(num_samples, num_labelled, gamma):
    # Initialize the semi-supervised model
    model = SemisupervisedModel(num_samples=num_samples, num_labelled=num_labelled, gamma=gamma)
    H, g = model.H, model.g
    # Solve the problem using np.linalg.solve
    y_pred = -np.linalg.solve(H, g)
    # Plot the clusters formed by the solution
    model.plot_clusters(np.sign(y_pred))


# This is the main function of the script
if __name__ == "__main__":
    # The function plot_solution is called with parameters num_samples=400, num_labelled=40, and gamma=0.5
    gradient_descent_convergence(num_samples=2000, num_labelled=200, gamma=0.5)