import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    # Сортируем признаки и цели вместе
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    # Уникальные значения признака (убираем дубликаты)
    unique_features = np.unique(sorted_features)

    # Если признак константный, возвращаем значения по умолчанию
    if len(unique_features) == 1:
        thresholds = np.array([])
        ginis = np.array([])
        threshold_best = None
        gini_best = -np.inf
        return thresholds, ginis, threshold_best, gini_best

    # Вычисляем пороги как средние между соседними уникальными значениями
    thresholds = (unique_features[:-1] + unique_features[1:]) / 2

    # Векторизованное вычисление критерия Джини для всех порогов
    n_total = len(target_vector)

    # Для каждого порога вычисляем статистики слева и справа
    ginis = []
    valid_thresholds = []

    for threshold in thresholds:
        # Разделяем на левую и правую части
        left_mask = sorted_features <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        # Пропускаем пороги, приводящие к пустым поддеревьям
        if n_left == 0 or n_right == 0:
            continue

        # Вычисляем доли классов в левой части
        left_targets = sorted_targets[left_mask]
        p1_left = np.mean(left_targets == 1)
        p0_left = 1 - p1_left
        H_left = 1 - p1_left ** 2 - p0_left ** 2

        # Вычисляем доли классов в правой части
        right_targets = sorted_targets[right_mask]
        p1_right = np.mean(right_targets == 1)
        p0_right = 1 - p1_right
        H_right = 1 - p1_right ** 2 - p0_right ** 2

        # Вычисляем критерий Джини
        gini = - (n_left / n_total) * H_left - (n_right / n_total) * H_right
        ginis.append(gini)
        valid_thresholds.append(threshold)

    if not ginis:  # Если нет валидных порогов
        thresholds = np.array([])
        ginis = np.array([])
        threshold_best = None
        gini_best = -np.inf
        return thresholds, ginis, threshold_best, gini_best

    ginis = np.array(ginis)
    valid_thresholds = np.array(valid_thresholds)

    # Находим лучший порог (при одинаковых gini берем минимальный threshold)
    best_idx = np.argmax(ginis)

    # Если есть несколько с одинаковым gini, берем первый (минимальный threshold)
    max_gini = ginis[best_idx]
    best_candidates = valid_thresholds[ginis == max_gini]
    threshold_best = np.min(best_candidates)
    gini_best = max_gini

    return valid_thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y != sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / current_click
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature]))
            else:
                raise ValueError

            if len(feature_vector) == 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "Categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        pass

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
