"""
A file for generating the graphs which will end up in the report.
"""

from plotting import (
    get_metric,
    get_metric_obj,
    get_multi_experiment_metric,
    variable_value_lines,
)


def main():
    # Every experiment is run using configs, and the data is saved in the /logs/automatic/ directory.
    # A directory which stores multiple of these experiment directories, is considered a multi-config
    # experiment. We can check the data of an experiment using several functions.
    metrics, description = get_metric_obj("MNIST_grid_test/b=0zs=4")
    print(f"Loaded metrics object:\n{description}\n with the following data:")
    print(metrics.archived_metrics.keys())
    # The above loads the metric object, which can be used to look into the data. The description is
    # a string which shows the config used to create that data. Notice that the experiment_name
    # parameter is part of a path, with two elements. This indicates that we are looking at an
    # experiment which is part of a multi-config experiment. Below that we check which metrics were
    # stored, so we can decided what we want to plot.

    # We can load specific data like this:
    losses_list: list = get_metric("MNIST_grid_test/b=0zs=4", "loss")
    # Or across a multi-config experiment like this:
    losses: dict[str, list] = get_multi_experiment_metric("MNIST_grid_test", "loss")

    # Now we have a list of data. What does this represent?
    # If there are specific metric keys prexifed by "train" in the data, then the ones without any
    # prefix are test set results. Otherwise, they are always training results. These results are
    # per batch. If you want per epoch you should adjust them using:
    #
    # from plotting import mean
    # [mean(i) for i in itertools.batched(losses, number_of_batches)]
    #
    # Now you can plot to your hearts desire!
    #
    # Look at some of the functions in the plotting.py file for inspiration.

    # Below we see how the variable_value_lines function works.
    # The get_multi_experiment_metric creates a dictionary with the sub-experiment names as keys,
    # and the metric values as values. However, we might want to influence the ordering of these
    # keys which is done through the 'order' keyword. We also adjust the keys, and make sure that
    # we only plot those keys which end with "zs=4".
    order = ["b=0", "b=0.1", "b=0.5", "b=1", "b=2", "b=5"]
    variable_value_lines(
        {k.split("z")[0]: v for k, v in losses.items() if k.endswith("zs=4")},
        title="ELBO loss",
        x_label="epochs",
        y_label="ELBO loss",
        path="latest_loss",
        order=order,
    )
    # This plot is automatically saved under logs/manual (what a choice, right?)
