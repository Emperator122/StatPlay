import math
import statistics as default_stat
import numpy as np
from scipy import stats
import pandas as pd


def statistics_test():
    default_stat_set = [15, 13, 11, 18, 10, 8, 20, 7, 8, 12, 15, 16, 13, 14, 14, 19, 7, 8, 11, 12, 15, 16, 13, 5, 11,
                        19, 18, 9, 6, 15]
    # Вычислим значения, которые будут использоваться далее
    st_dev_val = default_stat.stdev(default_stat_set)  # Стандартное отклонение
    max_val = max(default_stat_set)  # Максимум
    min_val = min(default_stat_set)  # Минимум
    # Вывод результатов
    print("Стандартный модуль статистики")
    print("Среднее:\t%s" % default_stat.mean(default_stat_set))
    print("Медиана:\t%s" % default_stat.median(default_stat_set))
    print("Стандартная ошибка:\t%s" % (st_dev_val/math.sqrt(len(default_stat_set))))
    print("Мода:\t%s" % default_stat.mode(default_stat_set))
    print("Стандартное отклонение: %s" % st_dev_val)
    print("Дисперсия:\t%s" % default_stat.variance(default_stat_set))
    print("Интервал:\t%s" % (max_val-min_val))
    print("Минимум:\t%s" % min_val)
    print("Максимум:\t%s" % max_val)
    print("Счет:\t%s" % len(default_stat_set))
    print("Сумма:\t%s" % sum(default_stat_set))


def numpy_test():
    numpy_stat_set_0 = [15, 13, 11, 18, 10, 8, 20, 7, 8, 12, 15, 16, 13, 14, 14, 19, 7, 8, 11, 12, 15, 16, 13, 5, 11,
                        19, 18, 9, 6, 15]
    numpy_stat_set = np.array(numpy_stat_set_0)  # Создадим numpy массив
    # Вычислим значения, которые будут использоваться далее
    st_dev_val = numpy_stat_set.std()  # Стандартное отклонение
    max_val = np.max(numpy_stat_set)  # Максимум
    min_val = np.min(numpy_stat_set)  # Минимум
    # Вывод
    print("numpy статистика")
    print("Среднее:\t%s" % numpy_stat_set.mean())
    print("Медиана:\t%s" % np.median(numpy_stat_set))
    print("Стандартная ошибка:\t%s" % (st_dev_val / math.sqrt(len(numpy_stat_set))))
    print("Мода:\t%s" % int(stats.mode(numpy_stat_set)[0]))
    print("Стандартное отклонение:\t%s" % st_dev_val)
    print("Дисперсия:\t%s" % numpy_stat_set.var())
    print("Интервал:\t%s" % (max_val - min_val))
    print("Минимум:\t%s" % min_val)
    print("Максимум:\t%s" % max_val)
    print("Счет:\t%s" % len(numpy_stat_set))
    print("Сумма:\t%s" % numpy_stat_set.sum())


def scipy_test():
    scipy_stat_set = [15, 13, 11, 18, 10, 8, 20, 7, 8, 12, 15, 16, 13, 14, 14, 19, 7, 8, 11, 12, 15, 16, 13, 5, 11,
                      19, 18, 9, 6, 15]
    print("scipy статистика")
    print(stats.describe(scipy_stat_set))


def pandas_test():
    # Считываем данные из csv
    dataset = pd.read_csv("py_stat.csv", index_col=0)
    # Выводим результаты
    print("pandas статистика\n")
    print("Результат метода describe")
    print(dataset.describe())
    print("\nМода")
    print(dataset.mode())
    print("\nМедиана")
    print(dataset.median())
    print("\nДисперсия")
    print(dataset.var())
    print("\nСумма")
    print(dataset.sum())


# Запускаем тестовые функции
statistics_test()
print()
numpy_test()
print()
scipy_test()
print()
pandas_test()
