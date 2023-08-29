import math
import random
import sys

import graphviz
from matplotlib import pyplot as plt
from sklearn import tree
from prettytable import PrettyTable as pt
"""Data description
NPG: Number of times pregnant
PGL: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
DIA: Diastolic blood pressure (mm Hg)
TSF: Triceps skin fold thickness (mm)
INS: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index
DPF: Diabetes pedigree function
AGE: Age (years)
Diabetic: Class variable 
"""
class Model:
    # count = 0
    #python3 DecisionTree.py C:\Users\Lutfi\Desktop\DiabetesData.csv "Diabetic"
    def __init__(self, name: object, data: dict, training_split: int, test_split: int, seed=10) -> object:
        self.name = name
        sum = training_split + test_split
        train_ratio = training_split / sum
        test_ratio = test_split / sum
        self.training_split_prec = f'{train_ratio * 100}%'
        self.test_split_prec = f'{test_ratio * 100}%'
        data_size = len(data[list(data.keys())[0]])
        training_size = math.ceil(train_ratio * data_size)
        test_size = data_size - training_size
        self.training_split = {}
        self.test_split = {}
        self.make_training_test_splits(self.training_split, self.test_split, training_size, test_size, data, data_size, seed)
        self.test_table = pt()
        self.training_table = pt()
        self.make_data_table(self.training_split, self.training_table)
        self.make_data_table(self.test_split, self.test_table)
        # Model.count += 1
        # self.num = Model.count

    def make_training_test_splits(self, training_split, test_split, training_size, test_size, data, data_size, seed):
        indexes_pool = list(range(data_size))
        random.seed(seed)
        random.shuffle(indexes_pool, )  # shuffling the indexes of the dataset
        training_indexes = indexes_pool[:training_size]
        test_indexes = indexes_pool[training_size:]

        for key in data:
            key_list = data[key]

            training_split[key] = []
            for ix in training_indexes:
                training_split[key].append(key_list[ix])

            test_split[key] = []
            for ix in test_indexes:
                test_split[key].append(key_list[ix])

    def make_data_table(self, data, data_table):
        # data_table = pt()
        for key in data.keys():
            data_table.add_column(key, data[key])

    def print_test_data(self):
        print(self.test_table)

    def print_training_data(self):
        print(self.training_table)


class Trainer:
    def __init__(self, dataset_file, target_attr):
        self.target_attr = target_attr
        file = open(dataset_file)
        file_content = file.read().split('\n')
        # for cont in file_content: print(cont)
        self.data = self.process_file_content(file_content)
        self.data_table = pt()
        if self.data == False:
            print("Error: Header has to be at the top of the dataset file!")
            exit()
        else:
            self.make_data_table(self.data)
        self.classes = self.count_classes(self.data, target_attr)
        self.models = []
        self.graphs = {}

    def add_model(self, name, training_split, test_split, seed):
        model = Model(name, self.data, training_split, test_split, seed)
        for m in self.models:
            if m.name == model.name:
                # print("Error: Model with the same name already exists!")
                return False
        self.models.append(model)
        self.graphs[model.name] = None
        return True

    def process_file_content(self, file_content):
        data = {}
        attributes = []
        for i, line in enumerate(file_content):
            line = line.strip()
            if i == 0:
                if line == "":
                    return False
                attributes = line.split(",")
                for attr in attributes:
                    attr = attr.strip()
                    data[attr] = []
            elif line == "":
                # return i
                continue
            else:
                values = line.split(",")
                for val, attr in zip(values, attributes):
                    val = val.strip()
                    attr = attr.strip()
                    data[attr].append(val)
        return data

    def make_data_table(self, data):
        for key in data.keys():
            self.data_table.add_column(key, data[key])

    def print_data(self):
        print(self.data_table)

    def calc_sdv(self, data, attr):
        data_len = len(data[attr]) - 1
        mean = self.calc_mean(data, attr)
        sum = 0
        for val in data[attr]:
            sum += (float(val) - mean) ** 2
        return round(math.sqrt(sum / data_len), 3)

    def calc_mean(self, data, attr):
        data_len = len(data[attr])
        sum = 0
        for val in data[attr]:
            sum += float(val)
        return round(sum / data_len, 3)

    def calc_median(self, data, attr):
        data_len = len(data[attr])
        sorted_data = sorted(data[attr])
        if data_len % 2 == 0:
            return round((float(sorted_data[int(data_len / 2)]) + float(sorted_data[int(data_len / 2) - 1])) / 2, 3)
        else:
            return round(float(sorted_data[int(data_len / 2)]), 3)

    def calc_min(self, data, attr):
        min = float(data[attr][0])
        for val in data[attr]:
            if float(val) < min:
                min = float(val)
        return min

    def calc_max(self, data, attr):
        max = float(data[attr][0])
        for val in data[attr]:
            if float(val) > max:
                max = float(val)
        return max

    def calc_percentile(self, data, attr, percentile):
        data_len = len(data[attr]) - 1
        sorted_data = sorted(data[attr])
        index = (percentile / 100) * data_len + 1
        fraction = index - int(index)
        p = float(sorted_data[int(index)]) + fraction * (
                float(sorted_data[int(index) + 1]) - float(sorted_data[int(index)]))
        return round(p, 3)

    def count_classes(self, data, attr):
        classes = {}
        for val in data[attr]:
            if val not in classes.keys():
                classes[val] = 1
            else:
                classes[val] += 1
        return classes

    def make_stats(self, data):
        stats = {}
        col_labels = ["Mean", "Median", "Standard Deviation", "Minimum", "Maximum", "25th Percentile",
                      "50th Percentile", "75th Percentile"]
        stats["stats"] = col_labels
        for attr in data.keys():
            stats[attr] = {}
            stats[attr]["Mean"] = self.calc_mean(data, attr)
            stats[attr]["Median"] = self.calc_median(data, attr)
            stats[attr]["Standard Deviation"] = self.calc_sdv(data, attr)
            stats[attr]["Minimum"] = self.calc_min(data, attr)
            stats[attr]["Maximum"] = self.calc_max(data, attr)
            stats[attr]["25th Percentile"] = self.calc_percentile(data, attr, 25)
            stats[attr]["50th Percentile"] = self.calc_percentile(data, attr, 50)
            stats[attr]["75th Percentile"] = self.calc_percentile(data, attr, 75)
            # print(stats[attr].values())

        stats_table = pt()
        stats_table.add_column("Stats", col_labels)
        for attr in data.keys():
            stats_table.add_column(attr, list(stats[attr].values()))
        print(stats_table)
        for key in self.classes:
            print(f"Class {key}: {round(self.classes[key] / len(data[attr]), 5) * 100}%")

    def plot_feature(self, data, attr):
        class_0 = 0
        class_1 = 0
        for ix in range(len(data[attr])):
            if data["Diabetic"][ix] == "0":
                class_0 += 1
            else:
                class_1 += 1
        plt.bar(["0", "1"], [class_0, class_1])
        plt.show()

    def get_model(self, name):
        for model in self.models:
            if model.name == name:
                return model
        return False

#extracts the features (X) and the corresponding target values (Y) 
#appends each feature vector to X while appending the target value to Y
def output_data_as_list(data, target_attr):
    data_len = len(data[list(data.keys())[0]])
    X = []
    Y = []
    for i in range(data_len):
        row = []
        for key in data.keys():
            if key == target_attr:
                Y.append(data[key][i])
            else:
                row.append(data[key][i])
        X.append(row)
    return X, Y


#training and evaluating a decision tree model.
#min_samples_leaf: minimum number of samples required to be at a leaf node.
def run_tree(trainer, model, min_samples_leaf=40):
    #
    x, y = output_data_as_list(model.training_split, trainer.target_attr)
    clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=min_samples_leaf)
    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True,
                                    class_names=["Non-Diabetic", "Diabetic"],
                                    feature_names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    graph = graphviz.Source(dot_data)
    graph.render(f"Diabetic Decision Tree Model {model.name}")
    #extract the test data (z and targets) from the model's test split
    z, targets = output_data_as_list(model.test_split, trainer.target_attr)
    
    preds = clf.predict(z)

    comp = []
    right = 0
    total = 0
    #evaluates the accuracy of the model by comparing each predicted value with the corresponding target value. 
    for pred, target in zip(preds, targets):
        if pred == target:
            comp.append(True)
            right += 1

        else:
            comp.append(False)
        total += 1

    # print(comp)
    print(f'Model {model.name} accuracy: {round(right / total * 100,5)}%')
    return graph

def create_model(trainer, seed=10):
    name = input("Enter the name of the model: ")
    training_split = input("Enter the training split: ")
    test_split = input("Enter the test split: ")
    try:
        training_split = int(training_split)
        test_split = int(test_split)
    except ValueError:
        print("Invalid split value")
        return
    if trainer.add_model(name, training_split, test_split, seed):
        print(f"Model {name} added successfully!")
    else:
        print(f"Model {name} already exists!")


    # model.clf = run_tree(model)
    # return model
def reshuffle_models_data(trainer):
    seed = input("Enter the seed: ")
    try:
        seed = int(seed)
    except ValueError:
        print("Invalid seed value")
        return
    new_models = []
    for model in trainer.models:
        name = model.name
        training_split = len(model.training_split[trainer.target_attr])
        test_split = len(model.test_split[trainer.target_attr])
        model = Model(name,trainer.data, training_split, test_split, seed)
        new_models.append(model)
    trainer.models = new_models
def print_menu():
    # print(f"Dataset fetched from {filename}")
    print("Select an option:")
    print("1. Print Dataset")
    print("2. Print Stats")
    print("3. Plot Class Distribution")
    print("4. Create Model")
    print("5. View Models")
    print("6. View Models Data")
    print("7. Test Models")
    print("8. Reshuffle Model Data")
    print("9. View Models Graphs")
    print("10. Delete Model")
    print("11. Exit")
    print("---------------------------")



def fetch_model():
    name = input("Enter the name of the model: ")
    model = trainer.get_model(name)
    if model == False:
        print("Model not found!")
        return False
    return model
def view_graphs(trainer):
    model = fetch_model()
    if model == False:
        return
    graph = trainer.graphs[model.name]
    graph.view()
def list_models(trainer):
    if len(trainer.models) == 0:
        print("No models found!")
        return
    for m in trainer.models:
        print(f'name: {m.name}, training split: {len(m.training_split[trainer.target_attr])}, test split: {len(m.test_split[trainer.target_attr])}')

def view_model_data():
    model = fetch_model()
    if model == False:
        return
    print("Training split:")
    model.print_training_data()
    print("Test split:")
    model.print_test_data()

def test_models(trainer):
    for m in trainer.models:
        g = run_tree(trainer, m)
        trainer.graphs[m.name] = g


def delete_model(trainer):
    model = fetch_model()
    if model == False:
        return
    trainer.models.remove(model)
    print(f"Model {model.name} deleted successfully!")


def main():
    while True:
        print_menu()
        choice = input("Enter your choice: ")

        def switch(option):
            switcher = {
                1: lambda: trainer.print_data(),
                2: lambda: trainer.make_stats(trainer.data),
                3: lambda: trainer.plot_feature(trainer.data, trainer.target_attr),
                4: lambda: create_model(trainer),
                5 : lambda : list_models(trainer),
                6: lambda : view_model_data(),
                7: lambda: test_models(trainer),
                8: lambda: reshuffle_models_data(trainer),
                9: lambda: view_graphs(trainer),
                10: lambda: delete_model(trainer),
                11: lambda: sys.exit()
            }
            try:
                option = int(option)
                if option > 11:
                    print("Invalid option!")
                    return
            except ValueError:
                print("Invalid input!")
                return
            func = switcher.get(option)
            func()

        switch(choice)


if __name__ == "__main__":
    filename = sys.argv[1]
    attr = sys.argv[2]
    # filename = "diabetesData.csv"
    # attr = "Diabetic"
    trainer = Trainer(filename, attr)
    main()

# trainer.make_stats(trainer.data)
# trainer.print_data()

# m1 = Model(trainer.data, 50, 50)
# m2 = Model(trainer.data, 70, 30)
# m1.print_test_data()
# # m1.print_training_data()
# m2.print_test_data()


# clf = run_tree(m3)
# test_on_all_dataset(trainer_external, clf)
# run_tree(m1)
# run_tree(m2)

# trainer.make_stats(trainer.data)
#
#
# trainer.plot_feature(trainer.data,trainer.target_attr)
