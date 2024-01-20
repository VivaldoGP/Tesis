import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from matplotlib.dates import DateFormatter, MonthLocator


def simple_line_plot(data: DataFrame | Series, x_data: str, y_data: str,
                     title: str, x_label: str, y_label: str,
                     export: bool = False, export_path: str = None):

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('_mpl-gallery')

    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.yaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)

    ax.plot(data[x_data], data[y_data], color='g', linewidth=1.5)
    ax.set(xlabel=x_label, ylabel=y_label, title=title)

    plt.xticks(rotation=45)

    if export:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

    return True


def multiple_line_plot(data: DataFrame, x_data: str, y_data: list,
                       x_label: str, y_label: list, export: bool = False, export_path: str = None,
                       subtitle: str = None):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(subtitle)
    plt.style.use('_mpl-gallery')

    ax1.xaxis.set_major_locator(MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax1.yaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax1.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax1.plot(data[x_data], data[y_data[0]], color='r', linewidth=1.5)
    ax1.set(xlabel=x_label, ylabel=y_label[0])

    ax2.xaxis.set_major_locator(MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax2.yaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax2.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax2.plot(data[x_data], data[y_data[1]], color='g', linewidth=1.5)
    ax2.set(xlabel=x_label, ylabel=y_label[1])

    ax3.xaxis.set_major_locator(MonthLocator(interval=1))
    ax3.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax3.yaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax3.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax3.plot(data[x_data], data[y_data[2]], color='b', linewidth=1.5)
    ax3.set(xlabel=x_label, ylabel=y_label[2])

    plt.xticks(rotation=0)

    if export:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

    return True