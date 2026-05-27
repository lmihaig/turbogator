from plot_flop_decomposition import generate_flop_decomposition
from plot_performance import generate_performance_plot
from plot_perf_bars import generate_perf_bars
from plot_roofline import generate_roofline_plot
from plot_speedup_bars import generate_speedup_bars


def main():
    generate_performance_plot()
    generate_roofline_plot()
    generate_perf_bars()
    generate_speedup_bars()
    generate_flop_decomposition()


if __name__ == "__main__":
    main()
