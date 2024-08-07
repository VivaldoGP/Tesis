import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from matplotlib.dates import DateFormatter, MonthLocator, WeekdayLocator
import pandas as pd
from datetime import datetime


def simple_line_plot(data: DataFrame | Series, x_data: str, y_data: str,
                     title: str, x_label: str, y_label: str,
                     export: bool = False, export_path: str = None):
    fig, ax = plt.subplots()
    plt.style.use('_mpl-gallery')

    ax.xaxis.set_major_locator(WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    ax.yaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)

    ax.plot(data[x_data], data[y_data], color='g', linewidth=1.5, marker='o', markersize=3, markerfacecolor='r')
    ax.set(xlabel=x_label, ylabel=y_label, title=title)

    plt.xticks(rotation=45)

    if export:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

    return True


def two_line_plot(data1: DataFrame, data2: DataFrame, x_data: str, y_data: list,
                  title: str, x_label: str, y_label: list, export: bool = False, export_path: str = None,
                  label1: str = 'Ndvi promedio', label2: str = 'ndvi promedio 2'):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.style.use('_mpl-gallery')

    ax1.xaxis.set_major_locator(MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax1.yaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax1.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)

    ax1.plot(data1[x_data], data1[y_data[0]], color='r', linewidth=1.5, label=label1)

    ax1.plot(data2[x_data], data2[y_data[1]], color='b', linewidth=1.5, label=label2)

    ax1.set(xlabel=x_label, ylabel=y_label, title=title)
    ax1.legend()
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


def precipitation_plot(data: DataFrame | Series, x_data: str, y_data: str, title: str, x_label: str, y_label: str,
                       export: bool = False, export_path: str = None, legend: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('_mpl-gallery')

    ax.xaxis.set_major_locator(MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.yaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)

    ax.plot(data[x_data], data[y_data], label=legend, color='b', alpha=0.7, linestyle='-', linewidth=0.8)
    ax.set(xlabel=x_label, ylabel=y_label, title=title)

    plt.xticks(rotation=45)
    ax.legend()

    if export:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

    return True


def vi_vs_climate_plot(vi_data: DataFrame | Series, climate_data: DataFrame | Series,
                       vi_data_columns: list, climate_data_columns: list, title: str, x_label: str, y_label: list,
                       export: bool = False, export_path: str = None, x_label_type: str = 'Fecha',
                       vi_color: str = 'yellowgreen', climate_color: str = 'royalblue'):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # plt.style.use('_mpl-gallery')

    if x_label_type == 'Fecha':
        ax1.xaxis.set_major_locator(MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    elif x_label_type == 'dias':
        ax1.yaxis.grid(True, which='major', linestyle='--', color=vi_color, alpha=.5, zorder=0)
        ax1.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
        ax1.tick_params(axis='y', labelcolor=vi_color)
        ax1.plot(vi_data[vi_data_columns[0]], vi_data[vi_data_columns[1]], color=vi_color,
                 linestyle='-', linewidth=1, alpha=0.7)
        ax1.set(xlabel=x_label, ylabel=y_label[0])

        ax2 = ax1.twinx()
        ax2.xaxis.grid(True, which='major', linestyle='-', color='grey', alpha=0.25, zorder=0)
        ax2.yaxis.grid(True, which='major', linestyle='--', color=climate_color, alpha=.5, zorder=0)
        ax2.tick_params(axis='y', labelcolor=climate_color)
        ax2.plot(climate_data[climate_data_columns[0]], climate_data[climate_data_columns[1]], color=climate_color,
                 linestyle='-', linewidth=1, label='Temperatura')
        ax2.set_ylabel(y_label[1])

        plt.xticks(rotation=45)
        plt.title(title)
        plt.show()

        return True


def multi_column(data: DataFrame | Series, x_data: str, y_data: str,
                 title: str, x_label: str, y_label: list, matrix: list):
    rows, cols = matrix[0], matrix[1]
    fig, ax = plt.subplots(rows, cols, figsize=(12, 8))

    for i in range(rows):
        for j in range(cols):
            ax[i, j].plot(data[x_data], data[y_data], color='g', linewidth=1.5)
            ax[i, j].set(xlabel=x_label, ylabel=y_label)

    plt.show()
    return True


def poly_degree_aic(data: DataFrame, x_data: str, y_data: list,
                    title: str, x_label: str, y_label: list,
                    export: bool = False, export_path: str = None):
    fig, ax2 = plt.subplots(figsize=(16, 8))
    plt.style.use('_mpl-gallery')

    ax2.xaxis.grid(True, which='major', linestyle='-', color='grey', alpha=0.25, zorder=0)
    ax2.yaxis.grid(True, which='major', linestyle='--', color='#BE3455', alpha=.5, zorder=0)
    # ax2.tick_params(axis='y', labelcolor='red')
    ax2.bar(data[x_data], data[y_data[1]], color='#FFBE98', alpha=1, width=0.3,
            linewidth=1.5, edgecolor='black', label='R2')
    ax2.set(xlabel=x_label, ylabel=y_label[1])
    ax2.legend(loc='upper left')

    ax1 = ax2.twinx()

    ax1.yaxis.grid(True, which='major', linestyle='--', color='#BE3455', alpha=.25, zorder=0)
    ax1.xaxis.grid(True, which='major', linestyle='--', color='grey', alpha=.25, zorder=0)
    ax1.plot(data[x_data], data[y_data[0]], color='#BE3455', linewidth=1, linestyle='-', alpha=0.8,
             label='AIC')
    ax1.set(xlabel=x_label, ylabel=y_label[0])
    ax1.legend(loc='upper right')

    plt.xticks(rotation=45)
    plt.title(title)

    if export:
        plt.savefig(export_path, dpi=100, bbox_inches='tight')
        # plt.show()



def poly_degree_plot(data: DataFrame, compare_column: str):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    columns = data.columns[-9:]

    # Iterar sobre las columnas del DataFrame y crear una subfigura para cada una
    for i in range(3):
        for j in range(3):
            col = columns[i * 3 + j]
            axs[i, j].plot(pd.to_datetime(data['Fecha']), data[col])
            axs[i, j].plot(pd.to_datetime(data['Fecha']), data[compare_column], linestyle='--', color='red')
            axs[i, j].set_title(col)
            axs[i, j].set_xlabel('Fecha')
            axs[i, j].set_ylabel('Valor')

    plt.tight_layout()

    # Mostrar la figura
    plt.show()
    pass


def column_adjust_plot(data: DataFrame, columnas_plot: list, titulo: str):
    rows = len(columnas_plot) // 2
    cols = 2

    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

    fig.suptitle(titulo)

    for i in range(rows):
        for j in range(cols):
            col = columnas_plot[i * 2 + j]
            axs[i, j].xaxis.set_major_locator(MonthLocator(interval=4))
            axs[i, j].xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            axs[i, j].plot(pd.to_datetime(data['Fecha']), data[col], label=col, lw=1.5)
            axs[i, j].legend()
            axs[i, j].set_title(col.upper())
            axs[i, j].set_xlabel('Fecha')
            axs[i, j].set_ylabel('Valor')

    plt.tight_layout()
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    pass


def scatter_plot(data: DataFrame, x_data: str, y_data: str, title: str, x_label: str, y_label: str,
                 export: bool = False):

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('_mpl-gallery')

    ax.scatter(data[x_data], data[y_data], color='g', alpha=0.7, marker='o', s=10)
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    plt.legend()
    plt.show()
    pass


def time_series(data: DataFrame, x_data: str, y_data: str, title: str, x_label: str,
                y_label: str, export: bool = False, export_path: str = None,
                zoom: bool = False, start_date: datetime = None, end_date: datetime = None,
                y_lim: tuple = None, annotate: bool = False, nube: datetime = None, color: str = 'b'):

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('_mpl-gallery')

    ax.plot(data[x_data], data[y_data], color=color, linewidth=1.5, markersize=3)
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    plt.xticks(rotation=45)

    if annotate:
        plt.annotate('Nube', xy=(nube, 0.5), xytext=(nube, 0.6), arrowprops=dict(facecolor='black', shrink=0.05))

    if zoom:
        ax.set_xlim(start_date, end_date)
        ax.set_ylim(y_lim[0], y_lim[1])

    if export:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
